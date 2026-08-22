"""Harness configuration and the named ablation modes.

Every stage is independently switchable. That is not stylistic: if stages cannot
be disabled one at a time, an improvement cannot be attributed to a mechanism,
and the experiment produces a number instead of a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class HarnessConfig:
    # --- stages ---------------------------------------------------------
    context_selection: bool = False
    policy: bool = False
    confidence_gate: bool = False
    memory: bool = False
    output_controls: bool = False        # length + question restraint + rerank
    validator: bool = False
    repetition_control: bool = False

    # --- context --------------------------------------------------------
    context_turns: int = 6               # turns kept when selection is on
    context_token_budget: int = 256      # the model supports 1024; we test less

    # --- policy ---------------------------------------------------------
    n_candidates: int = 4                # deterministic branch width for rerank
    oracle_policy: bool = False          # research mode: policy supplied externally

    # --- confidence -----------------------------------------------------
    min_top1: float = 0.34               # below this, prefer a conservative act
    min_margin: float = 0.06             # top1 - top2

    # --- length ---------------------------------------------------------
    max_new_tokens: int = 20

    # --- memory ---------------------------------------------------------
    memory_slots: int = 8

    # Reject a candidate reply already used this conversation. Measured on
    # v0.3.1-r004: 40% of replies inside a conversation were verbatim repeats of
    # an earlier one ("that sounds rough" three turns running). Tracking what has
    # already been said is exact bookkeeping and belongs in the harness, not in
    # the 6.7M model's weights.
    avoid_repeats: bool = False
    repeat_window: int = 4

    # --- Phase 3 steering interfaces (mutually comparable, one variable each) --
    steer: str = "none"          # none | restrict | bias | hidden
    steer_steps: int = 1         # how many opening tokens the interface touches
    steer_scale: float = 1.0     # logit-bias multiplier
    steer_alpha: float = 0.0     # hidden-state push, relative to hidden norm
    prefix_map: str = "artifacts/harness/prefixes.json"

    # Mode G: tiny learned policy classifier over the frozen model's hidden state.
    policy_head: str | None = None

    name: str = "custom"

    def with_(self, **kw) -> "HarnessConfig":
        return replace(self, **kw)


# Ablation ladder. Each mode adds exactly one mechanism to the one above it, so
# a difference between adjacent rows is attributable to that mechanism.
MODES: dict[str, HarnessConfig] = {
    "A_RAW": HarnessConfig(name="A_RAW"),
    # Repetition control ALONE on top of raw, so its contribution is separable
    # from every other mechanism.
    "R_NOREPEAT": HarnessConfig(name="R_NOREPEAT", avoid_repeats=True,
                                output_controls=True, repetition_control=True),
    "B_CONTEXT": HarnessConfig(name="B_CONTEXT", context_selection=True),
    "C_POLICY": HarnessConfig(name="C_POLICY", context_selection=True, policy=True),
    "D_POLICY_CONFIDENCE": HarnessConfig(
        name="D_POLICY_CONFIDENCE", context_selection=True, policy=True,
        confidence_gate=True),
    "E_POLICY_MEMORY": HarnessConfig(
        name="E_POLICY_MEMORY", context_selection=True, policy=True,
        confidence_gate=True, memory=True),
    "F_FULL_HARNESS": HarnessConfig(
        name="F_FULL_HARNESS", context_selection=True, policy=True,
        confidence_gate=True, memory=True, output_controls=True,
        validator=True, repetition_control=True),
    # The learned head. Same pipeline as F, but the policy comes from a 4,883-
    # parameter linear probe instead of same-model exemplar scoring, which was
    # measured at 19.2%.
    "G_LEARNED_HEAD": HarnessConfig(
        name="G_LEARNED_HEAD", context_selection=True, policy=True,
        confidence_gate=True, memory=True, output_controls=True,
        validator=True, repetition_control=True,
        policy_head="artifacts/harness/policy_head.pt"),
    # Research-only upper bound: the correct policy is supplied, so whatever
    # remains is language realisation rather than policy selection.
    "ORACLE_POLICY": HarnessConfig(
        name="ORACLE_POLICY", context_selection=True, policy=True,
        confidence_gate=False, memory=True, output_controls=True,
        validator=True, repetition_control=True, oracle_policy=True),
}

# Phase 3. Each row changes exactly ONE thing versus G_LEARNED_HEAD (the steering
# interface), and every mechanism has an oracle twin so classifier error can be
# separated from steering error.
def _steer(name, **kw):
    return HarnessConfig(
        name=name, context_selection=True, policy=True, confidence_gate=True,
        memory=True, output_controls=True, validator=True,
        repetition_control=True, **kw)


for _n, _kw in {
    "S2_RESTRICT":  dict(steer="restrict", policy_head="artifacts/harness/policy_head.pt"),
    "S3_BIAS":      dict(steer="bias", steer_steps=2, steer_scale=2.0,
                         policy_head="artifacts/harness/policy_head.pt"),
    "S4_HIDDEN":    dict(steer="hidden", steer_steps=2, steer_alpha=0.25,
                         policy_head="artifacts/harness/policy_head.pt"),
    "S4_HIDDEN_a10": dict(steer="hidden", steer_steps=2, steer_alpha=0.10,
                          policy_head="artifacts/harness/policy_head.pt"),
    "S4_HIDDEN_a50": dict(steer="hidden", steer_steps=2, steer_alpha=0.50,
                          policy_head="artifacts/harness/policy_head.pt"),
    "S4_HIDDEN_a100": dict(steer="hidden", steer_steps=4, steer_alpha=1.00,
                           policy_head="artifacts/harness/policy_head.pt"),
}.items():
    MODES[_n] = _steer(_n, **_kw)
    MODES["ORACLE_" + _n] = _steer("ORACLE_" + _n, oracle_policy=True,
                                   **{k: v for k, v in _kw.items()
                                      if k != "policy_head"})

# Inferred-policy versions at the strengths that worked under oracle.
for _a in (0.60, 0.80, 1.30):
    MODES[f"S4_INF_a{int(_a * 100):03d}"] = _steer(
        f"S4_INF_a{int(_a * 100):03d}", steer="hidden", steer_steps=2,
        steer_alpha=_a, policy_head="artifacts/harness/policy_head.pt")

# Steering strength sweep under a correct policy, to separate "the interface is
# weak" from "the strength was wrong".
for _a, _st in ((0.10, 2), (0.25, 1), (0.25, 4), (0.40, 2), (0.60, 2),
                (0.80, 2), (1.00, 2), (1.30, 2), (0.80, 4), (0.80, 1)):
    MODES[f"ORACLE_S4_a{int(_a * 100):03d}_s{_st}"] = _steer(
        f"ORACLE_S4_a{int(_a * 100):03d}_s{_st}", oracle_policy=True,
        steer="hidden", steer_steps=_st, steer_alpha=_a)
