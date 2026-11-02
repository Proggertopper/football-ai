# Football AI

An experimental **computer-vision and rule-based control pipeline** for offline football-game sandbox experiments, with EA Sports FC 26-specific action mappings.

The project explores how screen observations can be converted into tracked objects, an estimated match state and scored controller actions. It combines YOLO-based detection with geometric heuristics, temporal filtering and a virtual gamepad interface.

**Status: research prototype.** The offline logic can be evaluated without a running game or trained detector. Live operation requires additional software, compatible model weights and environment-specific calibration. This is not a trained end-to-end reinforcement-learning agent or a verified competitive gameplay system.

**Project scope:** development is currently paused at the experimental stage. I do not plan to turn this repository into a turnkey autonomous gameplay bot, because that could facilitate cheating in online matches. The repository is shared to document the engineering approach and offline experiments, not to distribute a competitive advantage.

## Main components

| Module | Responsibility |
| --- | --- |
| [`capture/`](capture/) | Screen capture through MSS or Windows DXCam |
| [`vision/`](vision/) | Ultralytics YOLO inference, jersey-color team classification, IoU-based tracking and geometry |
| [`state/`](state/) | Match-state estimation, possession and pressure heuristics, passing lanes and short-horizon prediction |
| [`tactics/`](tactics/) | Tactical finite-state machine and mode evaluation |
| [`decision/`](decision/) | Action candidates, heuristic scoring, cooldowns, hysteresis and a separately implemented safety layer |
| [`control/`](control/) | Action execution, queued controller inputs and virtual Xbox gamepad integration |
| [`observe/`](observe/) | Debug overlays, performance metrics, event logging, telemetry and replay-recorder utilities |
| [`evaluate/`](evaluate/) | Synthetic fixtures and scenario-based evaluation of state estimation and decision logic |
| [`tests/`](tests/) | Automated tests for geometry, state estimation, scoring, decisions and evaluation scenarios |
| [`config/`](config/) | YAML configuration, defaults and override loading |

The live loop follows capture → detection → tracking → state estimation → action selection → controller execution. Tactical mode information and telemetry are calculated alongside this loop. See **Current limitations** for incomplete integrations.

## Technology stack

- **Python**, dataclasses, type annotations and modular interfaces
- **NumPy** and **OpenCV** for frame processing and visualization
- **Ultralytics YOLO / PyTorch** for object detection
- **MSS / DXCam** for screen capture
- **vgamepad** for virtual controller output
- **PyYAML** for configuration and **pytest** for tests
- Optional **TensorRT** export for a compatible NVIDIA/CUDA environment

## Start with offline evaluation

Run commands from the repository root. Python 3.10+ is needed for the syntax used by the project; the offline checks below were verified with Python 3.12.

### 1. Clone and create an environment

```bash
git clone https://github.com/Proggertopper/football-ai.git
cd football-ai
python -m venv .venv
```

Activate the environment using the command for your terminal:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS shell, for offline checks | `source .venv/bin/activate` |

### 2. Install the minimal offline dependencies

```bash
python -m pip install numpy PyYAML pytest
```

This is an explicit minimal setup, not a version-locked environment. A complete base dependency manifest is not yet included in the repository.

### 3. Run tests and scenarios

```bash
python -m pytest tests -q
python main.py --mode eval
```

At the initial repository revision `f891070`, these commands produced **77 passing tests** and **7/7 passing evaluation scenarios**. These results exercise synthetic data and do not measure live detection accuracy, gameplay quality or end-to-end latency.

Use the explicit `tests/` directory: the root-level `test.py` is a model-inspection script that loads `models/best.pt`, not part of the lightweight offline suite.

## Live sandbox setup

Live screen capture and virtual-controller operation target **Windows**. They have not been validated by the offline checks described above.

Before attempting live mode:

1. Install the additional dependencies appropriate for your environment: `opencv-python`, `mss`, `keyboard`, `ultralytics`, and `vgamepad`; add `dxcam` when using that capture backend. Configure the virtual-controller driver/backend required by your installation.
2. For GPU inference, install a compatible PyTorch/CUDA stack. TensorRT is optional and requires a compatible export/runtime setup.
3. Supply compatible trained detection weights. Model files are excluded from Git; no trained checkpoint, training dataset or reproducible training recipe is currently distributed here.
4. Match the model's labels to `vision.class_map` in [`config/default.yaml`](config/default.yaml). A generic YOLO checkpoint is not a replacement for the expected game-specific classes.
5. Calibrate capture region, dimensions, world-coordinate settings, team colors and controller mappings for your local setup.

**Dependency-file note:** [`requirements-gpu.txt`](requirements-gpu.txt) currently starts with `S# Optional...`. Remove the stray `S` so the line begins with `#` before using that file with pip. It is an optional GPU list, not a complete list of runtime dependencies.

### Configure a local override

The default configuration selects DXCam, CUDA and `models/best.engine`. For a non-TensorRT experiment, create a local YAML override, for example `config.local.yaml`:

```yaml
capture:
  backend: mss

vision:
  model_path: models/best.pt
  device: cpu
  half_precision: false
```

This changes the backend and inference device only; it does not supply model weights or guarantee real-time performance. Review the other defaults before running.

```bash
python main.py --mode snapshot --config config.local.yaml
python main.py --mode live --config config.local.yaml
```

The snapshot command captures the configured screen region, saving to `logs/snapshot.jpg` by default. Live mode initializes the pipeline, waits briefly and starts with controller execution inactive.

| Control | Behavior |
| --- | --- |
| **F1** | Enable controller execution |
| **F2** | Pause controller execution and cancel queued inputs |
| **Q** in the debug overlay | Exit when the OpenCV overlay is enabled |
| **Ctrl+C** | Interrupt execution and trigger cleanup |

The F1/F2 hotkeys require the `keyboard` package. If `vgamepad` is unavailable, the gamepad interface uses a no-op backend; decision logs alone do not prove that inputs reach the game.

## Current limitations

- **Replay playback is a stub.** `--mode replay --video ...` does not yet process recorded video.
- **Replay recording is not wired into the live frame loop.** `ReplayRecorder` exists, but the current loop does not call `record_frame()`. Many committed replay files are empty.
- **The safety layer is bypassed by `DecisionEngine`.** Its implementation and tests exist, but its filtering is not applied by the current decision path.
- **Detection loss does not automatically pause controller execution.** Live execution is primarily gated by F1/F2. State estimation also overrides pause/wait flags; unattended use is not appropriate.
- **Some integrations are partial.** The tactical FSM feeds mode observation/telemetry, and an FC26 mechanics object is instantiated, but their presence does not imply all configured behaviors affect live action selection.
- **Evaluation covers a limited set of synthetic situations.** Several scenarios accept broad sets of actions; passing them is not evidence of optimal tactical behavior.
- **Setup is not yet reproducible end to end.** Runtime dependencies are not locked, model provenance/training details are missing, and live behavior depends on hardware and calibration.

## Unfinished work and development boundaries

The limitations above describe the actual implementation, not a promise of future features. The project is being retained as an experimental portfolio repository; completing an autonomous gameplay system is outside its current scope.

The following would be appropriate directions if educational work resumes, but are **not currently implemented or scheduled**:

- Reproducible offline dependency setup and CI.
- Video analysis and replay evaluation without controller output.
- Documented regression fixtures and stricter synthetic scenarios.
- Startup validation, controller-release safeguards and clearer diagnostic tools.

Turnkey online-match automation, anti-cheat evasion and competitive exploitation are outside the intended development scope. This is a project-policy boundary, not a technical enforcement mechanism: the existing code still contains live capture and gamepad-control components. Missing replay integration, disabled safety filtering and other unfinished pieces should not be mistaken for deliberate anti-cheating protections.

## Intended use

For local, offline experimentation and learning about perception, state estimation and decision pipelines. Use only where automation is permitted. This project does not include anti-cheat bypasses and should not be used to automate online competitive play.

Not affiliated with or endorsed by EA Sports. Game names and third-party technologies belong to their respective owners.

## Author

**Oleksandr Kosynskyi**

- [GitHub](https://github.com/Proggertopper)
- [LinkedIn](https://www.linkedin.com/in/oleksandr-kosynskyi-java/)
