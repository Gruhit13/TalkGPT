from datasets import load_dataset
import numpy as np

dataset = load_dataset("mythicinfinity/libritts_r", name="dev", split="dev.clean").select(range(10))
print(dataset)

for item in dataset:
    for key, value in item.items():
        print(f"{key} - {value}")
    break