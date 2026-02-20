import os, torch, torch.nn as nn
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pattern_torch.pt')

class TinyPatternNet(nn.Module):
    def __init__(self, in_features=10, hidden=64, classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, hidden), nn.ReLU(),
            nn.Linear(hidden, classes)
        )
    def forward(self, x):
        return self.net(x)

_model = TinyPatternNet().to(DEVICE)
_model.eval()
_loaded = False
_classes = [
    'DoubleTop','DoubleBottom','HeadShoulders','InvHeadShoulders',
    'TriangleAsc','TriangleDesc','Flag','Pennant','WedgeRise','WedgeFall',
    'SupportRes','None'
]

def _ensure_loaded():
    global _loaded
    if _loaded: return
    if os.path.exists(WEIGHTS_PATH):
        sd = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        _model.load_state_dict(sd)
    _loaded = True

@torch.no_grad()
def infer_patterns_torch(x: np.ndarray):
    _ensure_loaded()
    if x.ndim != 2: raise ValueError('expected [T,F]')
    xt = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
    xt = xt.transpose(0,1).unsqueeze(0)
    logits = _model(xt)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    best_idx = int(np.argmax(probs))
    return {
        'backend': 'torch',
        'class': _classes[best_idx],
        'probs': { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
    }
