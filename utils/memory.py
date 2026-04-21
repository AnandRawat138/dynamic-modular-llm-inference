import torch
import psutil
from pynvml import *

def get_ram_usage():
    return psutil.virtual_memory().used / (1024**3)

def get_gpu_usage():
    if not torch.cuda.is_available():
        return 0
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024**3)