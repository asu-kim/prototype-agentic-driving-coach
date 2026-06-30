import time
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = nn.Sequential(
    nn.Linear(4096, 8192),
    nn.ReLU(),
    nn.Linear(8192, 8192),
    nn.ReLU(),
    nn.Linear(8192, 4096),
).to(device)

model.eval()

x = torch.randn(128, 4096, device=device)

print("Running GPU inference load. Press Ctrl+C to stop.")

try:
    with torch.no_grad():
        while True:
            start = time.perf_counter()
            y = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            print(f"Inference time: {ms:.2f} ms")
except KeyboardInterrupt:
    print("\nStopped.")