import torch
import GPUtil


def select_gpu():
    device_ids = GPUtil.getAvailable(order = 'memory')
    if not device_ids:
        print("No available GPUs")
        return None
    else:
        print(f"Selected GPU: {device_ids[0]}")
        return device_ids[0]