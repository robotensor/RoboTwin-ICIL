# RoboTwin-ICIL

A one-demonstration in-context imitation learning benchmark built on
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).

[v0.1.0](https://github.com/robotensor/RoboTwin-ICIL/releases/tag/v0.1.0) ·
[Release notes](docs/releases/v0.1.0.md) ·
[Training dataset](https://huggingface.co/datasets/robotensor/robotwin-icil-aloha-clean)

https://github.com/user-attachments/assets/cb239294-4a33-4289-b0ff-b8e23f202c56

## Benchmark

1. Generate a scene and record one successful demonstration from RoboTwin's scripted expert.
2. Rebuild the identical initial scene and verify its fingerprint.
3. Give the demonstration to a frozen policy, execute its actions, and score with RoboTwin's
   success checker.

The policy receives camera images and robot state, without privileged scene or task metadata.
Training happens outside the evaluator. Expert generation failures are excluded from policy
scores; scene mismatches are harness errors. This protocol measures imitation from the same
initial scene.

The standard evaluation covers **50 tasks**, **100 episodes per task per setting**, and the
Aloha-AgileX robot. **Clean (Easy)** and **randomized (Hard)** settings use RoboTwin's evaluation
seed stream and are scored separately by the mean of the 50 task success rates. Official scores
require seed group 0 and every episode scored; shorter runs are provisional.

## Quick start

Install the simulator and its assets on an NVIDIA GPU machine:

```bash
git clone --recurse-submodules https://github.com/robotensor/RoboTwin-ICIL.git
cd RoboTwin-ICIL
bash scripts/install_robotwin.sh
```

Activate the installed `robotwin` conda environment, then run a replay smoke test:

```bash
conda activate robotwin
robotwin-icil eval --policy replay --task click_bell --episodes 1 --seed 42 \
  --run-dir runs/smoke
robotwin-icil report runs/smoke
```

Replay checks the harness using the expert's recorded actions. See the
[installation guide](docs/install.md) for GPU support and troubleshooting.

## Evaluate a policy

Use an importable `module:Class` adapter implementing `ICILPolicy`. Its lifecycle is
`reset` → `set_demonstration` → `act`; actions can be returned in chunks. Local adapters work
without the private `icil-policy` package. See the [policy guide](docs/policies.md).

```bash
robotwin-icil standard-eval --policy your_adapter:Policy \
  --policy-arg checkpoint=/absolute/model --run-dir runs/standard/model
robotwin-icil standard-report runs/standard/model
```

The command evaluates both settings and resumes when repeated. For a provisional interface check,
use `--setting clean --episodes-per-task 1`. See the
[standard benchmark guide](docs/standard-benchmark.md) for scoring and reporting requirements.

## Training data

The [clean Aloha dataset](https://huggingface.co/datasets/robotensor/robotwin-icil-aloha-clean)
contains **2,500 trajectories** across all 50 tasks, with original HDF5 data and aligned head and
wrist camera videos. All trajectories are training data; evaluation demonstrations are generated
in simulation. There is no stored train/validation/test split.

Dataset tools run without the simulator. See the
[dataset instructions](docs/standard-benchmark.md#training-data) for download, verification,
and loading.

## Competition integration

The optional [competition plugin](competition/README.md) provides deterministic duel units,
shared prompts, and result validation for the Robotensor ICIL orchestrator. Served policies and
competition integration require its private `icil-policy` package; setup is covered in the
plugin guide.

## Documentation

- [Simulator installation](docs/install.md)
- [Policy adapters and served policies](docs/policies.md)
- [Standard evaluation and dataset tools](docs/standard-benchmark.md)
- [Competition plugin](competition/README.md)

## Citation and credits

Cite RoboTwin-ICIL using [CITATION.cff](CITATION.cff) and report the benchmark commit used in your
experiments. Thank you to the [RoboTwin team](https://github.com/RoboTwin-Platform/RoboTwin) for
the simulation, tasks, experts, and success checks that make this benchmark possible.

RoboTwin-ICIL is licensed under [Apache-2.0](LICENSE). The RoboTwin submodule retains its MIT license.
