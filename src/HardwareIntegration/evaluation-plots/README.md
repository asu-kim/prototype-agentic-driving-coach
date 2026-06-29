# Evaluation plotting scripts

This directory contains one plotting script per evaluation criterion.

## Setup

```bash
python3 -m pip install -r src/HardwareIntegration/evaluation-plots/requirements.txt
```

## Plots

```bash
python3 src/HardwareIntegration/evaluation-plots/plot_network_latency.py
python3 src/HardwareIntegration/evaluation-plots/plot_gpu_inference_latency.py
python3 src/HardwareIntegration/evaluation-plots/plot_model_timing_variation.py
python3 src/HardwareIntegration/evaluation-plots/plot_deterministic_replay.py --reference-dir src/HardwareIntegration/data-collection/logs-trace-deeksha-normal --candidate-dir fed-gen/HardwareIntegratedADCReplay/logs-trace
```

By default, figures and summary CSVs are written to:

```text
src/HardwareIntegration/evaluation-plots/out
```

Each script accepts `--data-root` and `--out-dir`. Use `--help` on any script for the full list of options.
