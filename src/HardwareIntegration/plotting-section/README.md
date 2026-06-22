# Section 3 Plotting: Understanding Latencies

This directory contains plotting code for the paper section on LF tracing, latency, reaction triggers, and replay-vs-collected behavior.

## Files

- `plot_latency_section.py`: generates figures and summary tables from trace CSVs.
- `figures/`: output directory created by the script.

## Default Inputs

The script defaults to the current collected/replay files:

- collected car inputs: `src/HardwareIntegration/data-collection/logs-trace-2/car_inputs.csv`
- replay car inputs: `fed-gen/HardwareIntegratedADCReplay/logs-trace/car_inputs.csv`
- collected execution trace: `src/HardwareIntegration/data-collection/logs-trace-2/federate_execution_times.csv`
- replay execution trace: `fed-gen/HardwareIntegratedADCReplay/logs-trace/federate_execution_times.csv`
- current execution trace: `fed-gen/HardwareIntegratedADC/logs-trace/federate_execution_times.csv` (optional; included automatically when present)

After you regenerate aligned traces, either overwrite those files or pass new paths with CLI flags.

## Run

From the repository root:

```bash
python3 src/HardwareIntegration/plotting-section-3/plot_latency_section.py
```

With explicit paths:

```bash
python3 src/HardwareIntegration/plotting-section-3/plot_latency_section.py \
  --original-car path/to/collected/car_inputs.csv \
  --replay-car path/to/replay/car_inputs.csv \
  --original-exec path/to/collected/federate_execution_times.csv \
  --replay-exec path/to/replay/federate_execution_times.csv \
  --current-exec path/to/current/federate_execution_times.csv \
  --out-dir src/HardwareIntegration/plotting-section-3/figures
```


