# Deterministic Prototyping and Testing Environment for Agentic CPS

A deterministic prototyping and testing environment for agentic cyber-physical systems, focused on a real-time agentic driving coach. The project combines Lingua Franca (LF) for deterministic execution, hardware-in-the-loop driver inputs, local LLM guidance, and MuJoCo-based vehicle simulation support.

The current hardware-integrated flow reads driver state, estimates vehicle/environment state, asks a local Llama model for driving-coach guidance, and can issue warning or actuation commands in a federated LF program.

## Repository Layout

- `src/HardwareIntegration/HardwareIntegratedADC.lf`: main hardware-integrated agentic driving coach LF program.
- `src/HardwareIntegration/SensorInputs.lf`: driver sensor input integration.
- `mujoco-py/`: MuJoCo LF support code included as a submodule.

## Setup

Clone the repository and initialize submodules:

```bash
git submodule update --init --recursive
```

Install the main prerequisites:

1. Install Lingua Franca and the LF compiler from the [LF installation guide](https://www.lf-lang.org/docs/installation/).
2. Install [Ollama](https://ollama.com/) for local Llama model inference.
3. Install the Python MuJoCo package:

```bash
python3 -m pip install mujoco
```

Pull the local Llama models used during experiments:

```bash
ollama pull llama3.2:1b
ollama pull llama3:8b
ollama pull llama3:70b
```

The active LF program uses `llama3:8b` in `HardwareIntegratedADC.lf`. Make sure Ollama is running before starting the federates.

## Build

Compile the main hardware-integrated LF program from the repository root:

```bash
lfc src/HardwareIntegration/HardwareIntegratedADC.lf
```

This generates the federated runtime under:

```bash
fed-gen/HardwareIntegratedADC/
```

Older stop-sign prototypes are available under `src/HardwareIntegration/Oldcodes/` if you need to rebuild earlier experiments.

## Run Federated Execution

After building, open separate terminals and run each process from the generated directory:

```bash
cd fed-gen/HardwareIntegratedADC
```

Start the RTI on the server:

```bash
./bin/RTI -n 5
```

Then start the federates. The driver federate should run on the Raspberry Pi or hardware input machine; the remaining federates usually run on the server.

```bash
./bin/federate__d
./bin/federate__c
./bin/federate__en
./bin/federate__adc
./bin/federate__sim
```

If you run the federates on different machines, update the federated reactor host address in `src/HardwareIntegration/HardwareIntegratedADC.lf` before compiling.

## Notes

- The system expects hardware sensor inputs for steering, acceleration, braking, head pose, and eye direction.
- The planner can provide spoken instructions through the local audio stack. Check the Piper and audio paths in `HardwareIntegratedADC.lf` if speech output does not play.
- MuJoCo simulation hooks are present in the LF source but are currently commented in the active hardware-integrated flow.

## Attribution

This work builds on the Lingua Franca MuJoCo integration by Edward A. Lee. See the original [MuJoCo LF Python repository](https://github.com/lf-lang/mujoco-py).
