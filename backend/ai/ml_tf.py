import os
import numpy as np

# TensorFlow is optional; the backend runs fine without it.
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pattern_tf.h5')

_classes = [
    'DoubleTop','DoubleBottom','HeadShoulders','InvHeadShoulders',
    'TriangleAsc','TriangleDesc','Flag','Pennant','WedgeRise','WedgeFall',
    'SupportRes','None'
]


def _build_model(T=256, F=10, classes=12):
    if tf is None:
        return None
    inp = tf.keras.Input(shape=(T,F))
    x = tf.keras.layers.Conv1D(64,5,padding='same',activation='relu')(inp)
    x = tf.keras.layers.Conv1D(64,5,padding='same',activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)


_model = _build_model()
_loaded = False


def _ensure_loaded():
    global _loaded, _model
    if tf is None or _model is None:
        _loaded = True
        return
    if _loaded:
        return
    if os.path.exists(WEIGHTS_PATH):
        _model.load_weights(WEIGHTS_PATH)
    _loaded = True


def infer_patterns_tf(x: np.ndarray):
    """Infer chart-pattern class probabilities using TF model if available.

    If TensorFlow isn't installed, returns a disabled payload rather than crashing.
    """
    if tf is None or _model is None:
        return {
            'backend': 'tf',
            'disabled': True,
            'reason': 'TensorFlow not installed',
            'class': 'None',
            'probs': {c: (1.0 if c=='None' else 0.0) for c in _classes},
        }

    _ensure_loaded()
    if x.ndim != 2:
        raise ValueError('expected [T,F]')
    x = x[np.newaxis, ...]
    probs = _model.predict(x, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    return {
        'backend': 'tf',
        'class': _classes[best_idx],
        'probs': { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
    }
