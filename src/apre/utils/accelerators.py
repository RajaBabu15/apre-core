import os
import warnings
import tensorflow as tf

try:
    import torch
except Exception:
    torch = None

warnings.filterwarnings("ignore")

def detect_accelerators():
    gpus = tf.config.list_physical_devices('GPU')
    has_cuda_gpu = len(gpus) > 0
    mps_available = False

    if torch is not None:
        try:
            mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_available = False

    return {
        "has_cuda_gpu": has_cuda_gpu,
        "mps_available": mps_available,
        "gpu_devices": gpus
    }

def configure_accelerators():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

    accel = detect_accelerators()
    print("Detected accelerators:", accel)

    if accel["has_cuda_gpu"]:
        for gpu in accel["gpu_devices"]:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    try:
        if accel["has_cuda_gpu"] or accel["mps_available"]:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("Enabled mixed precision.")
    except Exception as e:
        print("Mixed precision not enabled:", e)

    return accel

def set_random_seeds(seed=42):
    import numpy as np
    np.random.seed(seed)
    tf.random.set_seed(seed)
