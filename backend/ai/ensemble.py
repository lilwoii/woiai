import numpy as np
from .ml_torch import infer_patterns_torch
from .ml_tf import infer_patterns_tf

def build_features(bars):
    o = np.array([b['o'] for b in bars], dtype=np.float32)
    h = np.array([b['h'] for b in bars], dtype=np.float32)
    l = np.array([b['l'] for b in bars], dtype=np.float32)
    c = np.array([b['c'] for b in bars], dtype=np.float32)
    v = np.array([b.get('v',0) for b in bars], dtype=np.float32)
    eps = 1e-6
    delta = np.diff(c, prepend=c[0])
    body = np.abs(c-o)
    wtop = h - np.maximum(c,o)
    wbot = np.minimum(c,o) - l
    norm = (c - c.mean()) / (c.std() + eps)
    X = np.stack([o,h,l,c,v, delta, body, wtop, wbot, norm], axis=1)
    T = 256
    if len(X) >= T:
        X = X[-T:]
    else:
        pad = np.zeros((T - len(X), X.shape[1]), dtype=X.dtype)
        X = np.concatenate([pad, X], axis=0)
    return X

def infer_ensemble(bars):
    X = build_features(bars)
    out_t = infer_patterns_torch(X)
    out_tf = infer_patterns_tf(X)
    classes = list(out_t['probs'].keys())
    avg = {k: (out_t['probs'][k] + out_tf['probs'][k]) / 2.0 for k in classes}
    best = max(avg.items(), key=lambda kv: kv[1])[0]
    return {
        'class': best,
        'probs': avg,
        'backends': [out_t, out_tf]
    }
