import asyncio
import math
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from db import open_db
from providers import ohlc as ohlc_provider

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, Optional[float], Optional[float], Optional[float]], None]


def _get_gpu_metrics() -> Tuple[Optional[float], Optional[float]]:
    """Try to read GPU utilization / temp via nvidia-smi. Safe to fail."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        line = out.decode().strip().splitlines()[0]
        util_str, temp_str = [p.strip() for p in line.split(",")]
        return float(util_str), float(temp_str)
    except Exception:
        return None, None


class PriceDirectionDataset(Dataset):
    """
    Very simple direction dataset:

    Input: window of recent closes (length W)
    Label: 1 if next close > last close in window, else 0
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        # x: [N, window], y: [N]
        self.x = x
        self.y = y
        self.window = x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # return shape [1, window] for Conv1d
        return self.x[idx].unsqueeze(0), self.y[idx]


class PriceDirectionModel(nn.Module):
    def __init__(self, window: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, window]
        x = self.conv(x)
        x = self.head(x)
        return x


def _get_symbols(max_symbols: int = 8) -> List[str]:
    with open_db() as db:
        wl = db.get("watchlist", {})
    syms = sorted(wl.keys())
    if not syms:
        # fallback if watchlist empty
        syms = ["TSLA", "AAPL", "NVDA", "MSFT", "BTCUSD"]
    return syms[:max_symbols]


def _fetch_bars(sym: str, lookback: int = 1500, use_crypto: bool = False):
    if use_crypto:
        return asyncio.run(ohlc_provider.binance_klines(sym, "1m", lookback))
    else:
        return asyncio.run(ohlc_provider.alpaca_bars(sym, "1Min", lookback))


def build_dataset(
    window: int = 64,
    max_samples_per_symbol: int = 400,
    use_crypto: bool = False,
    log_fn: Optional[LogFn] = None,
) -> PriceDirectionDataset:
    """
    Build a tiny supervised dataset from recent OHLC data.

    - Uses watchlist symbols (or a default set if empty)
    - For each symbol: sliding windows of 'window' closes → up/down label
    """
    symbols = _get_symbols()
    seqs: List[List[float]] = []
    labels: List[int] = []

    for sym in symbols:
        if log_fn:
            log_fn(f"Loading bars for {sym}…")
        try:
            bars = _fetch_bars(sym, lookback=window + max_samples_per_symbol * 2, use_crypto=use_crypto)
        except Exception as e:
            if log_fn:
                log_fn(f"  ⚠️ failed to load bars for {sym}: {e}")
            continue

        closes = [float(b["c"]) for b in bars]
        if len(closes) <= window + 2:
            continue

        limit = min(len(closes) - 2, window + max_samples_per_symbol)
        for i in range(window, limit):
            window_slice = closes[i - window : i]
            # label: +1 if next bar up, else 0
            label = 1 if closes[i + 1] > closes[i] else 0
            seqs.append(window_slice)
            labels.append(label)

    if not seqs:
        raise RuntimeError("Not enough historical data to build training set.")

    x = torch.tensor(seqs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return PriceDirectionDataset(x, y)


def run_local_training_job(
    heavy: bool,
    log_fn: Optional[LogFn] = None,
    progress_fn: Optional[ProgressFn] = None,
) -> None:
    """
    Blocking training routine.

    This is called from a background thread. It:
      - builds a dataset from recent bars
      - trains a tiny conv model
      - periodically calls log_fn(...) and progress_fn(progress, eta, gpu%, temp)
      - saves weights to backend/ai/models/price_direction.pt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log_fn:
        log_fn(f"Using device: {device} (CUDA available={torch.cuda.is_available()})")

    # dataset
    dataset = build_dataset(
        window=64,
        max_samples_per_symbol=800 if heavy else 400,
        use_crypto=False,
        log_fn=log_fn,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    window = dataset.window

    model = PriceDirectionModel(window).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4 if heavy else 8e-4)

    epochs = 10 if heavy else 4
    total_steps = epochs * len(loader)
    step_idx = 0
    avg_step_time = 1.5  # crude ETA guess, seconds

    if log_fn:
        log_fn(f"Built dataset: {len(dataset)} samples across {len(loader)} batches.")
        log_fn(f"Training for {epochs} epochs (heavy={heavy}).")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in loader:
            step_start = time.time()
            step_idx += 1

            batch_x = batch_x.to(device)  # [B, 1, window]
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

            # progress (map into [0.05, 0.95])
            frac = step_idx / max(1, total_steps)
            progress = 0.05 + 0.9 * frac

            # ETA + GPU metrics
            dur = time.time() - step_start
            avg_step_time = 0.9 * avg_step_time + 0.1 * dur
            steps_left = max(0, total_steps - step_idx)
            eta = steps_left * avg_step_time

            gpu_load, gpu_temp = _get_gpu_metrics()

            if progress_fn:
                progress_fn(progress, eta, gpu_load, gpu_temp)

            if log_fn and step_idx % 10 == 0:
                msg = f"[epoch {epoch+1}/{epochs}] step {step_idx}/{total_steps} loss={loss.item():.4f}"
                if gpu_load is not None:
                    msg += f" (GPU {gpu_load:.0f}% @ {gpu_temp or 0:.0f}°C)"
                log_fn(msg)

        if log_fn:
            avg_loss = epoch_loss / max(1, len(loader))
            log_fn(f"Epoch {epoch+1}/{epochs} avg loss={avg_loss:.4f}")

    # save weights
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "price_direction.pt"
    torch.save(model.state_dict(), out_path)

    total_time = time.time() - start_time
    if log_fn:
        log_fn(f"✅ Local training complete in {total_time/60:.1f} min. Saved to {out_path}")
