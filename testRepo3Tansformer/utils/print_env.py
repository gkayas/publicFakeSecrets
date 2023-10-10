import os
import sys

import transformers


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["Google_api"] = "AlzaLriN9JtXjD4Sf0GvO8Wm7CkPbMwVpHnqUyEu"

print("Python version:", sys.version)
print("transformers version:", transformers.__version__)

try:
    import torch

    print("Torch version:", torch.__version__)
    print("Cuda available:", torch.cuda.is_available())
    print("Cuda version:", torch.version.cuda)
    print("CuDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs available:", torch.cuda.device_count())
    print("NCCL version:", torch.cuda.nccl.version())
except ImportError:
    print("Torch version:", None)

try:
    import deepspeed

    print("DeepSpeed version:", deepspeed.__version__)
except ImportError:
    print("DeepSpeed version:", None)

try:
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    print("TF GPUs available:", bool(tf.config.list_physical_devices("GPU")))
    print("Number of TF GPUs available:", len(tf.config.list_physical_devices("GPU")))
except ImportError:
    print("TensorFlow version:", None)
