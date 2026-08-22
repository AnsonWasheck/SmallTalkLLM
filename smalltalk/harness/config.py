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

    name: str = "custom"

    def with_(self, **kw) -> "HarnessConfig":
        return replace(self, **kw)


# Ablation ladder. Each mode adds exactly one mechanism to the one above it, so
# a difference between adjacent rows is attributable to that mechanism.
MODES: dict[str, HarnessConfig] = {
    "A_RAW": HarnessConfig(name="A_RAW"),
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
    # Research-only upper bound: the correct policy is supplied, so whatever
    # remains is language realisation rather than policy selection.
    "ORACLE_POLICY": HarnessConfig(
        name="ORACLE_POLICY", context_selection=True, policy=True,
        confidence_gate=False, memory=True, output_controls=True,
        validator=True, repetition_control=True, oracle_policy=True),
}
