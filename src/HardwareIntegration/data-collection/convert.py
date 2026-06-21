# #!/usr/bin/env python3
# import pandas as pd

# inp = "/home/raspdeeksha/deterministic-prototyping-agenticCPS/src/HardwareIntegration/data-collection/logs-trace-2/driver_control_inputs.csv"
# out = "driver_control_replay_values.csv"

# df = pd.read_csv(inp)
# df = df[df["event"] == "output_tick"].copy()

# keep = [
#     "logical_time_ns",
#     "logical_time_ms",
#     "steer",
#     "accelerator",
#     "brake",
# ]

# df[keep].to_csv(out, index=False)
# print(f"Saved {len(df)} rows to {out}")

#!/usr/bin/env python3
import pandas as pd

inp = "/home/raspdeeksha/deterministic-prototyping-agenticCPS/src/HardwareIntegration/data-collection/logs-trace-2/driver_monitor_inputs.csv"
out = "/home/raspdeeksha/deterministic-prototyping-agenticCPS/src/HardwareIntegration/data-collection/logs-trace-2/driver_monitor_replay_values.csv"

df = pd.read_csv(inp)

df = df[df["event"] == "output_tick"].copy()

keep = [
    "logical_time_ns",
    "logical_time_ms",
    "head",
    "eye",
]

df[keep].to_csv(out, index=False)

print(f"Saved {len(df)} rows to {out}")