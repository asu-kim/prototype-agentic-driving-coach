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
  --out-dir src/HardwareIntegration/plotting-section-3/figures
```


## LF Trace Artifacts (`.lft` and `trace_svg.html`)

The CSV plots summarize logged values, but the LF tracing output is also useful for the paper. The generated `.lft` files are binary LF trace logs, and files such as `trace_svg.html` are visualization artifacts produced by the LF trace tooling after the federated run.

Collect the current LF trace artifacts into this section folder:

```bash
python3 src/HardwareIntegration/plotting-section-3/collect_lf_trace_artifacts.py
```

If `trace_svg.html` is generated somewhere outside `fed-gen/HardwareIntegratedADCReplay`, pass it explicitly:

```bash
python3 src/HardwareIntegration/plotting-section-3/collect_lf_trace_artifacts.py   --extra path/to/trace_svg.html
```

Outputs are written to:

- `lf-trace-artifacts/manifest.csv`: source path, artifact type, size, and copied filename.
- `lf-trace-artifacts/index.html`: clickable index for opening `trace_svg.html` or other HTML views.
- copied `.lft` files: raw trace provenance for the exact federated run.

Suggested use in the paper:

- Include `trace_svg.html` or a screenshot from it as the **LF diagram with tracing**.
- Mention `.lft` files as the raw LF trace logs emitted by `target Python { tracing: true }`.
- Use the CSV plots in `figures/` for quantitative summaries, and use the LF trace HTML for visual scheduling/causality evidence.

## Generated Figures

- `01_reaction_trigger_timeline`: reaction vs tardy event timeline at `Car`.
- `02_reaction_trigger_counts`: count of ordinary and tardy `Car` reactions.
- `03_car_behavior_comparison`: velocity, accelerator, brake, and steer over logical time.
- `04_logical_time_by_row`: logical-time coverage by CSV row index.
- `05_physical_logical_lag`: physical elapsed time minus logical time at `Car`.
- `06_execution_time_p95_bars`: p95 reaction execution time by reactor/reaction.
- `07_execution_time_distribution`: execution-time histogram.
- `08_clock_period_histogram`: observed logical clock period/frequency from `Car` reaction tags.

Each figure is saved as both `.png` and `.svg` for paper editing.

## Generated Tables

- `summary_metrics.csv`: row counts, logical coverage, tardy counts, lag statistics.
- `reaction_trigger_counts.csv`: event counts by trace.
- `tardy_events.csv`: tardy rows with present flags and values.
- `execution_time_summary.csv`: count/mean/median/p95/max execution time by federate/reactor/reaction.
- `clock_period_summary.csv`: period and frequency summary.
- `clock_period_summary_samples.csv`: per-interval period samples.

## Notes for the Paper Section

Suggested mapping to the section outline:

- **LF diagram with tracing**: use the LF reactor diagram plus `01_reaction_trigger_timeline` to show where reactions fire.
- **How you trace**: describe CSV logging of logical time, physical time, tag time, microstep, present flags, and execution time.
- **Results from tracing**: use `summary_metrics.csv`, `reaction_trigger_counts.csv`, and `06_execution_time_p95_bars`.
- **Analysis**: use `03_car_behavior_comparison` and `05_physical_logical_lag` to discuss behavior and latency differences.
- **Clock period/frequency**: use `clock_period_summary.csv` and `08_clock_period_histogram`.
