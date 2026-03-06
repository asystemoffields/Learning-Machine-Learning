# Learning-Machine-Learning

An adventure space where I go way out of depth and totally over my head with ML.

## disco-torch

A PyTorch port of DeepMind's **Disco103** — the meta-learned reinforcement learning update rule from [*Discovering Reinforcement Learning Algorithms*](https://www.nature.com/articles/s41586-025-08828-z) (Nature, 2025).

### What is DiscoRL?

Instead of hand-crafted loss functions like PPO or GRPO, DiscoRL uses a small LSTM neural network (the "meta-network") that **generates loss targets** for RL agents. Given a rollout of agent experience (policy logits, rewards, advantages, auxiliary predictions), the meta-network outputs target distributions. The agent minimizes KL divergence between its outputs and these learned targets.

The Disco103 checkpoint (754,778 parameters) was meta-trained by DeepMind across thousands of Atari-like environments. It generalizes as a drop-in update rule for new tasks.

### Why a PyTorch port?

The original implementation uses JAX + Haiku. This port enables using Disco103 in PyTorch training pipelines — the eventual goal is bridging it to LLM fine-tuning (e.g. Qwen 4B).

### Project structure

```
disco_torch/               # The PyTorch package
  types.py                 # Dataclasses: UpdateRuleInputs, MetaNetInputOption, ValueOuts, etc.
  transforms.py            # Input transforms and construct_input()
  meta_net.py              # DiscoMetaNet — the full LSTM meta-network
  update_rule.py           # DiscoUpdateRule — meta-net + value computation + loss
  value_utils.py           # V-trace, TD-error, advantage estimation, Q-values
  utils.py                 # batch_lookup, signed_logp1, 2-hot encoding, EMA
  load_weights.py          # Maps JAX/Haiku NPZ keys → PyTorch modules

weights/
  disco_103.npz            # Pretrained checkpoint (42 params, 754K values) — git-ignored

scripts/
  inspect_disco103.py      # Print NPZ weight names and shapes
  validate_against_jax.py  # Numerical comparison: PyTorch vs JAX reference
  diagnose_*.py            # Debugging scripts for numerical divergence
```

### Meta-network architecture

```
Outer (per-trajectory):
  y_net           MLP [600 → 16 → 1]         Value prediction embedding
  z_net           MLP [600 → 16 → 1]         Auxiliary prediction embedding
  policy_net      Conv1dNet [9 → 16 → 2]     Action-conditional embedding
  trajectory_rnn  LSTM(27, 256)               Reverse-unrolled over trajectory
  state_gate      Linear(128 → 256)           Multiplicative gate from meta-RNN
  y_head / z_head Linear(256 → 600)           Loss targets for y and z
  pi_conv + head  Conv1dNet [258 → 16] → 1   Policy loss target (per action)

Meta-RNN (per-lifetime):
  Separate y/z/policy nets, input MLP(29 → 16), LSTMCell(16, 128)
```

### Status

**Working:** Full forward pass, weight loading (all 42 params), agent loss, value utilities, custom HaikuLSTMCell.

**Validated:** All outputs match JAX reference within floating-point precision (max_diff < 1.3e-06).

### Usage

```python
from disco_torch import DiscoUpdateRule, UpdateRuleInputs, load_disco103_weights

rule = DiscoUpdateRule()
load_disco103_weights(rule, "weights/disco_103.npz")

state = rule.meta_net.initial_meta_rnn_state()
meta_out, new_state = rule.meta_net(inputs, state)
# meta_out["pi"], meta_out["y"], meta_out["z"]
```

### Installation

```bash
pip install -e .
# For JAX validation: pip install disco_rl jax dm-haiku rlax
```

Download `disco_103.npz` from the [disco_rl repo](https://github.com/google-deepmind/disco_rl) and place in `weights/`.
