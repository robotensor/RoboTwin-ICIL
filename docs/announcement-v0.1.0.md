# Announcement draft — RoboTwin-ICIL v0.1.0

Editorial note: this is a draft for a public announcement. The GitHub repository was
private when the release was prepared; make the repository and release accessible to
the intended audience before posting the links below.

---

Can a frozen robot policy watch one successful demonstration and reproduce the
behaviour from the same starting scene?

We're releasing **RoboTwin-ICIL v0.1.0**, Robotensor's benchmark for
one-demonstration in-context imitation learning, built on **RoboTwin 2.0**.

For each episode, RoboTwin's expert generates a successful demonstration on demand.
The benchmark rebuilds the initial scene, verifies that it matches, and gives the
policy that single demonstration as context. The policy acts with frozen weights.
RoboTwin's own task success check scores the outcome.

The release includes:

- All 50 RoboTwin tasks, organized into seven manipulation skill categories, with
  task selection and a filter for the 26 strictly one-arm tasks.
- Aloha-AgileX and dual Franka-Panda support.
- A policy adapter interface, expert surveys, saved demonstration prompts,
  resumable evaluation records, and demonstration/rollout videos.
- Same Scene 1-Demo Success Rate, reported overall, by skill category and by task.

Our documented replay oracle scored **18/18 across nine tasks**, with every rebuilt
scene matching its demonstration's fingerprint. That validates the harness using
the expert's recorded trajectory; learned-policy results are the next step.

V0.1.0 asks a focused question about using a demonstration in context. Scene
generalization is outside this first release.

A big thank you to the **RoboTwin team** for openly releasing the simulation,
tasks, experts, and success checks that make this work possible.

If you're building policies that learn from demonstrations in context, we'd love
you to try the benchmark and contribute an adapter.

Repository: https://github.com/robotensor/RoboTwin-ICIL

Release: https://github.com/robotensor/RoboTwin-ICIL/releases/tag/v0.1.0

#Robotics #ImitationLearning #InContextLearning #RoboTwin
