"""Model adapters for the robotwin-icil benchmark, outside its core.

`robotwin-icil-policies` is a separate distribution. It depends on `robotwin-icil`; the core
never imports it. Each adapter converts the core's model-independent `Demonstration` and
`Observation` into what one model consumes, and the model's outputs back into RoboTwin actions,
and runs in that model's own environment (`scripts/install_policy_env.sh`).

`icil_policies.common` is what adapters share. It needs numpy and the core only, so it is
importable in the simulator environment and tested in CI.
"""
