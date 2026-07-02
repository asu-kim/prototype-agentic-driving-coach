# Prototype Agentic Driving Coach as a Distributed H/AITL CPS

This repository contains the prototype implementation agentic driving coach work. It uses [Lingua Franca](https://www.lf-lang.org/)/reactor MoC based approach to analyze timing latency.

The driving-coach scenario models a driver approaching an exit, changing lanes, slowing for an exit road, stopping, yielding, and moving ahead while the agent can issue warnings or actuation commands. The current codebase has two main LF entry points:

- `src/HardwareIntegration/HardwareIntegratedADC.lf`: live hardware-integrated execution with steering/throttle/brake input, webcam-based driver monitoring, MuJoCo simulation, Ollama LLM inference, and action planning.
- `src/HardwareIntegration/HardwareIntegratedADCReplay.lf`: deterministic replay execution from recorded CSV traces for latency, tardiness, and evaluation experiments.



## Repository Layout

- `src/HardwareIntegration/HardwareIntegratedADC.lf`: maian federated reactor.
- `src/HardwareIntegration/HardwareIntegratedADCReplay.lf`: main replay federated reactor.
- `src/HardwareIntegration/SensorInputs.lf`: combines steering wheel/gamepad input and webcam face detection.
- `src/HardwareIntegration/SensorReader.lf`: reads Linux input events and webcam frames.
- `src/HardwareIntegration/HeadPoseEstimation.lf`: wraps `face_detection.py` for head and eye state.
- `src/HardwareIntegration/Simulation.lf`: MuJoCo-based environment federate.
- `src/HardwareIntegration/evaluation*/`: logged data used by replay and timing experiments.
- `mujoco-py/`: MuJoCo LF support code included as a submodule.

## Prerequisites

Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```
```
cd mujoco-py/
git checkout demo
```

Install the main tools and Python packages:

```bash
python3 -m pip install mujoco rerun-sdk ollama evdev opencv-python numpy matplotlib
```

You also need:

- Lingua Franca and `lfc`.
- Ollama running locally for the live LLM federate.
- A Linux-compatible steering wheel/gamepad exposed through `evdev`.
- A webcam for driver head/eye monitoring.
- Piper and `aplay` if you want spoken planner instructions.

The live LLM federate currently calls:

```bash
ollama pull llama3.1:8b
```

## Build

Build the live hardware-integrated program:

```bash
lfc src/HardwareIntegration/HardwareIntegratedADC.lf
```

This generates:

```bash
fed-gen/HardwareIntegratedADC/
```

Build the replay program:

```bash
lfc src/HardwareIntegration/HardwareIntegratedADCReplay.lf
```

This generates:

```bash
fed-gen/HardwareIntegratedADCReplay/
```

Add the IP for the federated reactor at `<IP>>`. Update that address before compiling if the RTI host changes.

## Run Live Hardware Execution

Start Ollama first:

```bash
ollama serve
```

Then run the generated live program from separate terminals:

```bash
cd fed-gen/HardwareIntegratedADC
./bin/RTI -n 7
./bin/federate__d_control
./bin/federate__d_monitor
./bin/federate__c
./bin/federate__adc
./bin/federate__llm
./bin/federate__planner
./bin/federate__sim
```

Typical placement is to run `d_control` and `d_monitor` on the hardware input machine, and the remaining federates on the server. The live run writes logs under `logs-trace/` by default.

Useful log-path environment variables include:

- `DRIVER_CONTROL_LOG_PATH`
- `DRIVER_MONITOR_LOG_PATH`
- `ADC_INPUT_LOG_PATH`
- `LLM_BEHAVIOR_LOG_PATH`
- `PLANNER_LOG_PATH`
- `FEDERATE_EXECUTION_LOG_DIR`

## Run Replay Experiments

Replay runs from recorded traces instead of live hardware and live LLM calls. Use it to reproduce timing behavior, federate lag, tardy invocations, and paper-style network/GPU stress comparisons. Choose a trace by setting either `REPLAY_TRACE_NAME` or `REPLAY_TRACE_DIR`.

Example using a trace name under `src/HardwareIntegration/evaluation-replay`:

```bash
export REPLAY_TRACE_NAME=logs-trace-normal-assumption-network
lfc src/HardwareIntegration/HardwareIntegratedADCReplay.lf
cd fed-gen/HardwareIntegratedADCReplay
./bin/RTI -n 7
./bin/federate__d_control
./bin/federate__d_monitor
./bin/federate__c
./bin/federate__adc
./bin/federate__llm
./bin/federate__planner
./bin/federate__sim
```

## Evaluation Plots

Generate the evaluation figures with:

```bash
python3 src/HardwareIntegration/evaluation/<required plot file name>
```


## Notes

- `SensorReader.lf` only opens the wheel/gamepad in the `d_control` federate and only opens the webcam in the `d_monitor` federate.
- Steering values are mapped to `0` for left, `1` for straight, and `2` for right before entering the main program.
- Head and eye values are converted into recent driver attention history before the LLM prompt is generated.
- The planner combines LLM warnings with deterministic adaptive-cruise and lane-change logic.

## Attribution

This work builds on the Lingua Franca MuJoCo integration by Edward A. Lee. See the original [MuJoCo LF Python repository](https://github.com/lf-lang/mujoco-py).
