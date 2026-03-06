"""Numerical validation: compare PyTorch disco_torch against JAX disco_rl."""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from disco_rl.networks import meta_nets as jax_mn
from disco_rl.update_rules import disco as jax_disco
from disco_rl.agent import get_settings_disco
from disco_rl import types as jt

import torch
from disco_torch import DiscoUpdateRule, UpdateRuleInputs, load_disco103_weights

# ---- Shared data ----
np.random.seed(42)
T, B, A, PS = 5, 2, 4, 600

logits = np.random.randn(T+1, B, A).astype(np.float32)
beh_logits = np.random.randn(T+1, B, A).astype(np.float32)
y = np.random.randn(T+1, B, PS).astype(np.float32)
z = np.random.randn(T+1, B, A, PS).astype(np.float32)
actions = np.random.randint(0, A, (T+1, B)).astype(np.int32)
rewards = np.random.randn(T, B).astype(np.float32)
is_terminal = np.zeros((T, B), dtype=np.float32)
tgt_logits = np.random.randn(T+1, B, A).astype(np.float32)
tgt_y = np.random.randn(T+1, B, PS).astype(np.float32)
tgt_z = np.random.randn(T+1, B, A, PS).astype(np.float32)
v_scalar = np.random.randn(T+1, B).astype(np.float32)
adv = np.random.randn(T, B).astype(np.float32)
norm_adv = np.random.randn(T, B).astype(np.float32)
q = np.random.randn(T+1, B, A).astype(np.float32)
qv_adv = np.random.randn(T+1, B, A).astype(np.float32)
norm_qv_adv = np.random.randn(T+1, B, A).astype(np.float32)

NPZ = "weights/disco_103.npz"

# ---- JAX ----
settings = get_settings_disco()
net_config = settings.update_rule.net

def meta_net_fn(inputs, axis_name=None):
    return jax_mn.LSTM(**net_config)(inputs, axis_name=axis_name)

hk_fn = hk.transform_with_state(meta_net_fn)

jax_inputs = jt.UpdateRuleInputs(
    observations=jnp.zeros((T+1, B, 8)),
    actions=jnp.array(actions),
    rewards=jnp.array(rewards),
    is_terminal=jnp.array(is_terminal, dtype=jnp.bool_),
    agent_out={"logits": jnp.array(logits), "y": jnp.array(y), "z": jnp.array(z)},
    behaviour_agent_out={"logits": jnp.array(beh_logits), "y": jnp.array(y), "z": jnp.array(z)},
    extra_from_rule={
        "v_scalar": jnp.array(v_scalar), "adv": jnp.array(adv),
        "normalized_adv": jnp.array(norm_adv), "q": jnp.array(q),
        "qv_adv": jnp.array(qv_adv), "normalized_qv_adv": jnp.array(norm_qv_adv),
        "target_out": {"logits": jnp.array(tgt_logits), "y": jnp.array(tgt_y), "z": jnp.array(tgt_z)},
    },
)

rng = jax.random.PRNGKey(0)
jax_params, jax_state = hk_fn.init(rng, jax_inputs, axis_name=None)

# Replace init params with NPZ weights
npz_data = np.load(NPZ)
for mod in jax_params:
    for pname in jax_params[mod]:
        key = f"{mod}/{pname}"
        if key in npz_data:
            jax_params[mod][pname] = jnp.array(npz_data[key])

# Zero the RNN state
jax_state = jax.tree.map(jnp.zeros_like, jax_state)

jax_out, jax_new_state = hk_fn.apply(jax_params, jax_state, rng, jax_inputs, axis_name=None)

print("JAX:")
for k in ("pi", "y", "z", "meta_input_emb"):
    v = jax_out[k]
    print(f"  {k}: {v.shape}  mean={float(v.mean()):.8f}")

# ---- PyTorch ----
rule = DiscoUpdateRule()
load_disco103_weights(rule, NPZ)

pt_inputs = UpdateRuleInputs(
    observations=torch.zeros(T+1, B, 8),
    actions=torch.from_numpy(actions.astype(np.int64)),
    rewards=torch.from_numpy(rewards),
    is_terminal=torch.from_numpy(is_terminal),
    agent_out={"logits": torch.from_numpy(logits), "y": torch.from_numpy(y), "z": torch.from_numpy(z)},
    behaviour_agent_out={"logits": torch.from_numpy(beh_logits), "y": torch.from_numpy(y), "z": torch.from_numpy(z)},
)
pt_inputs.extra_from_rule = {
    "v_scalar": torch.from_numpy(v_scalar), "adv": torch.from_numpy(adv),
    "normalized_adv": torch.from_numpy(norm_adv), "q": torch.from_numpy(q),
    "qv_adv": torch.from_numpy(qv_adv), "normalized_qv_adv": torch.from_numpy(norm_qv_adv),
    "target_out": {"logits": torch.from_numpy(tgt_logits), "y": torch.from_numpy(tgt_y), "z": torch.from_numpy(tgt_z)},
}

state = rule.meta_net.initial_meta_rnn_state()
with torch.no_grad():
    pt_out, pt_new_state = rule.meta_net(pt_inputs, state)

print("\nPyTorch:")
for k in ("pi", "y", "z", "meta_input_emb"):
    v = pt_out[k]
    print(f"  {k}: {v.shape}  mean={v.mean():.8f}")

# ---- Compare ----
print("\n--- Comparison ---")
for k in ("pi", "y", "z", "meta_input_emb"):
    pt_v = pt_out[k].numpy()
    jax_v = np.array(jax_out[k])
    max_diff = np.max(np.abs(pt_v - jax_v))
    mean_diff = np.mean(np.abs(pt_v - jax_v))
    tag = "PASS" if max_diff < 1e-4 else ("CLOSE" if max_diff < 1e-2 else "FAIL")
    print(f"  {k:20s}  max={max_diff:.2e}  mean={mean_diff:.2e}  [{tag}]")

# Compare meta-RNN state
pt_h = pt_new_state[0].numpy()
jax_h = np.array(jax_new_state["lstm"]["meta_rnn_state"].hidden)
h_diff = np.max(np.abs(pt_h - jax_h))
print(f"  {'meta_rnn_h':20s}  max={h_diff:.2e}  [{('PASS' if h_diff < 1e-4 else 'CLOSE' if h_diff < 1e-2 else 'FAIL')}]")
