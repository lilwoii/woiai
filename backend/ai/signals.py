from typing import List, Tuple

def infer_signal(spark: List[float], heavy: bool = False) -> Tuple[str, float, str]:
    if not spark or len(spark) < 10:
        return 'neutral', 0.50, 'Insufficient data'
    last = spark[-1]
    first = spark[0]
    change = (last - first) / (first if first != 0 else 1)
    slope = sum((spark[i] - spark[i-1]) for i in range(1, len(spark))) / max(1, len(spark))
    if change > 0.01 and slope > 0:
        return 'buy', min(0.99, 0.60 + abs(change)), 'Uptrend momentum'
    if change < -0.01 and slope < 0:
        return 'sell', min(0.99, 0.60 + abs(change)), 'Downtrend momentum'
    return 'neutral', 0.55, 'Sideways/noise'
