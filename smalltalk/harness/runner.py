"""The harness pipeline.

    user message
      -> deterministic features
      -> conversation state
      -> memory (observe / retrieve)
      -> context selection
      -> constrained policy decision      (scored with the SAME model)
      -> confidence gate
      -> generation + policy-aware rerank (candidates from the SAME model)
      -> validation (at most one retry)
      -> state update
      -> reply

Every stage is switchable via HarnessConfig, and A_RAW is asserted by test to be
byte-identical to the unmodified engine, so any measured gain is attributable to
a mechanism rather than to incidental changes in prompting or decoding.

No stage ever emits text it authored. Policy exemplars are scoring probes; the
visible reply always comes from the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from pathlib import Path

from ..infer.generate import GenerationConfig, generate
from .config import HarnessConfig, MODES
from .confidence import score as score_confidence
from .context import select as select_context
from .features import extract
from .memory import Memory
from .policy import (BY_ID, CONSERVATIVE, CONSERVATIVE_CLOSING, LENGTH_TOKENS,
                     POLICIES, Policy, QuestionPolicy, consistency,
                     high_confidence_shortcut)
from .state import ConversationState
from .steering import PrefixMap, steered_generate
from .trace import Trace
from . import repetition
from .validator import validate


@dataclass
class Harness:
    model: object
    tokenizer: object
    cfg: HarnessConfig = field(default_factory=lambda: MODES["A_RAW"])
    gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(
        temperature=0.0, top_p=1.0, top_k=0, greedy=True,
        repetition_penalty=1.0, max_new_tokens=20, seed=0))
    history: list[dict] = field(default_factory=list)
    state: ConversationState = field(default_factory=ConversationState)
    memory: Memory = field(default_factory=Memory)
    _calls: int = 0
    _baselines: dict = field(default_factory=dict)
    _head: object = None
    _head_norm: object = None
    _prefixes: object = None
    _centroids: object = None

    def __post_init__(self) -> None:
        if self.cfg.steer != "none":
            self._prefixes = PrefixMap.load(self.cfg.prefix_map)
            cpath = Path(self.cfg.prefix_map).with_name("centroids.pt")
            if self.cfg.steer == "hidden" and cpath.exists():
                self._centroids = torch.load(cpath, map_location="cpu",
                                             weights_only=False)
        if self.cfg.policy_head:
            import torch as _t

            from .head import PolicyHead
            p = Path(self.cfg.policy_head)
            if p.exists():
                self._head = PolicyHead.load(p)
                self._head_norm = _t.load(p.with_suffix(".norm.pt"),
                                          map_location="cpu", weights_only=False)

    def _head_policy(self, prompt_ids: list[int], f) -> dict[str, float]:
        """Policy distribution from the learned probe: one extra forward pass."""
        from .head import POLICY_IDS, feature_vector, hidden_state

        h = hidden_state(self.model, prompt_ids, device=self.device)
        self._calls += 1
        if self._head.n_features:
            h = torch.cat([h, feature_vector(f)])
        h = (h.unsqueeze(0) - self._head_norm["mu"]) / self._head_norm["sd"]
        with torch.no_grad():
            probs = torch.softmax(self._head(h)[0], dim=-1)
        return {pid: float(pr) for pid, pr in zip(POLICY_IDS, probs)}

    # ---- plumbing ------------------------------------------------------
    def reset(self) -> None:
        self.history = []
        self.state = ConversationState()
        self.memory = Memory(slots=self.cfg.memory_slots)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _encode(self, messages: list[dict]) -> list[int]:
        ids, _ = self.tokenizer.encode_conversation(
            messages, add_bos=True, add_generation_prompt=True)
        return ids

    # ---- policy inference ---------------------------------------------
    @torch.no_grad()
    def _score_exemplar(self, prompt_ids: list[int], text: str) -> float:
        """Mean log P(text | prompt) under the unchanged model.

        Length-normalised so that short exemplars are not mechanically favoured;
        without normalisation every policy with a one-word exemplar would win.
        """
        cont = self.tokenizer.encode(text)
        if not cont:
            return -1e9
        ids = prompt_ids + cont
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(x)
        self._calls += 1
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        total = 0.0
        for i, tok in enumerate(cont):
            pos = len(prompt_ids) + i - 1
            total += float(logprobs[pos, tok])
        return total / len(cont)

    def _baseline(self, text: str) -> float:
        """Unconditional score of an exemplar, computed once and cached.

        Raw log P(exemplar | context) is dominated by the model's prior over
        common phrases: "i'm good, how about you?" is probable after almost
        anything, so the argmax collapsed onto one policy and classification
        accuracy measured 19.2%. Subtracting the context-free score turns this
        into pointwise mutual information -- how much THIS conversation raises
        the exemplar's probability -- which is the quantity we actually want.

        The baseline does not depend on the conversation, so it is computed once
        per exemplar and the correction is effectively free.
        """
        if text not in self._baselines:
            null = [self.tokenizer.bos_id, self.tokenizer.assistant_id]
            self._baselines[text] = self._score_exemplar(null, text)
        return self._baselines[text]

    def _infer_policy(self, prompt_ids: list[int]) -> dict[str, float]:
        scores = {}
        for p in POLICIES:
            scores[p.pid] = max(
                self._score_exemplar(prompt_ids, e) - self._baseline(e)
                for e in p.exemplars)
        t = torch.tensor(list(scores.values()))
        probs = torch.softmax(t, dim=0)
        return {pid: float(pr) for pid, pr in zip(scores, probs)}

    # ---- generation ----------------------------------------------------
    @torch.no_grad()
    def _candidates(self, prompt_ids: list[int], max_new: int, k: int) -> list[str]:
        """Deterministic k-best: branch on the top-k first tokens, then greedy.

        Sampling would make the harness non-reproducible, and beam search would
        collapse onto near-identical strings. Branching the first token gives
        genuinely distinct openings at a cost of k forward passes, and is exactly
        reproducible.
        """
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(x)
        self._calls += 1
        first = torch.topk(logits[0, -1].float(), k).indices.tolist()

        outs: list[str] = []
        for tok in first:
            if tok in (self.tokenizer.endofturn_id, self.tokenizer.eos_id,
                       self.tokenizer.pad_id):
                continue
            sub = generate(self.model, self.tokenizer, prompt_ids + [tok],
                           self.gen.with_(max_new_tokens=max(1, max_new - 1)),
                           device=self.device)
            self._calls += 1
            text = self.tokenizer.decode([tok] + sub).strip()
            if text and text not in outs:
                outs.append(text)
        return outs

    def _rerank(self, cands: list[str], policy: Policy | None,
                use_consistency: bool = True) -> str:
        """Pick the candidate most consistent with the chosen policy.

        Consistency only -- length class and question policy. The harness never
        rewrites or composes text; it chooses among things the model produced.
        """
        if policy is None:
            # No trusted policy: fall back to the model's own first choice,
            # rejecting only outright repetition.
            recent = [h["content"] for h in self.history if h["role"] == "assistant"]
            for c in cands:
                if not repetition.is_exact_repeat(c, recent) \
                        and not repetition.internal_loop(c):
                    return c
            return cands[0] if cands else ""
        cap = LENGTH_TOKENS[policy.length]
        recent = [h["content"] for h in self.history if h["role"] == "assistant"]

        def penalty(c: str) -> tuple:
            n = len(self.tokenizer.encode(c))
            over = max(0, n - cap)
            # Candidates arrive in the model's own top-k order. Ranking is the
            # last tiebreaker so that, absent a hard violation, the harness
            # defers to the model rather than substituting its own taste.
            rank = cands.index(c)
            # Negated so that higher consistency sorts earlier under `min`.
            match = -round(consistency(c, policy), 2) if use_consistency else 0.0
            is_q = c.strip().endswith("?")
            # Only NO_QUESTION is a constraint. PREFERRED was originally treated
            # as "required", which made the harness discard a correct greedy
            # answer to force a question: "i got the callback" -> "right?" and
            # "off to bed" -> "long day?". A preference must never override a
            # reply the model was already confident about.
            q_bad = policy.question is QuestionPolicy.NO_QUESTION and is_q
            rep = repetition.is_exact_repeat(c, recent) or repetition.internal_loop(c)
            return (rep, q_bad, over, match, rank)

        return min(cands, key=penalty) if cands else ""

    # ---- the turn ------------------------------------------------------
    def reply(self, user_text: str, *, oracle: Policy | None = None,
              trace: Trace | None = None) -> str:
        t0 = time.perf_counter()
        self._calls = 0
        tr = trace or Trace()
        tr.user, tr.mode = user_text, self.cfg.name

        f = extract(user_text,
                    prev_assistant_was_question=self.state.last_response_was_question,
                    consecutive_assistant_questions=self.state.consecutive_questions)
        tr.features = f.as_dict()

        self.history.append({"role": "user", "content": user_text.strip()})

        if self.cfg.memory:
            learned = self.memory.observe(user_text, self.state.turn_index)
            tr.memory_learned = [x.as_dict() for x in learned]
            retrieved = self.memory.retrieve(user_text)
            tr.memory_retrieved = [x.as_dict() for x in retrieved]
        else:
            retrieved = []

        hint = None
        if retrieved:
            hint = " ".join(f"{x.key.replace('_', ' ')}: {x.value}." for x in retrieved)

        if self.cfg.context_selection:
            sel = select_context(self.history, self.tokenizer,
                                 max_turns=self.cfg.context_turns,
                                 token_budget=self.cfg.context_token_budget,
                                 memory_hint=hint)
            messages, prompt_ids = sel.messages, self._encode(sel.messages)
            tr.context = sel.as_dict()
        else:
            messages = list(self.history)
            prompt_ids = self._encode(messages)[-self.gen.max_context:]
            tr.context = {"n_turns": len(messages), "n_tokens": len(prompt_ids),
                          "dropped": 0}

        # --- policy -----------------------------------------------------
        policy: Policy | None = None
        source = "none"
        if self.cfg.policy:
            if self.cfg.oracle_policy and oracle is not None:
                policy, source = oracle, "oracle"
            else:
                shortcut = high_confidence_shortcut(f)
                if self._head is not None:
                    probs = self._head_policy(prompt_ids, f)
                    inferred_source = "head"
                else:
                    probs = self._infer_policy(prompt_ids)
                    inferred_source = "model"
                tr.policy_scores = {k: round(v, 4) for k, v in probs.items()}
                conf = score_confidence(probs, min_top1=self.cfg.min_top1,
                                        min_margin=self.cfg.min_margin)
                tr.confidence = conf.as_dict()
                best = BY_ID[max(probs, key=probs.get)]
                if shortcut is not None:
                    policy, source = shortcut, "shortcut"
                elif self.cfg.confidence_gate and conf.status == "LOW":
                    # Genuinely conservative means DECLINING TO STEER, not
                    # substituting a clarifying question. Forcing "what do you
                    # mean?" onto an ambiguous distribution replaced correct
                    # replies with worse ones in the first ablation, because a
                    # flat distribution means the harness does not know better
                    # than the model -- so it should get out of the way.
                    policy, source = None, "abstain"
                else:
                    policy, source = best, inferred_source
        tr.policy = str(policy) if policy else None
        tr.policy_source = source

        if policy is not None and policy.action.value == "CLOSE":
            self.state.conversation_closing = True

        # --- generation --------------------------------------------------
        # Length control belongs to the policy stage: without it C/D/E compute a
        # policy and then ignore it, which made those rungs byte-identical to
        # A_RAW in the first ablation and told us nothing.
        max_new = self.cfg.max_new_tokens
        if policy is not None and self.cfg.policy:
            max_new = min(max_new, LENGTH_TOKENS[policy.length] + 4)

        if self.cfg.steer != "none" and policy is not None:
            # One steered greedy decode. The interface touches only the opening
            # tokens; everything after is ordinary autoregressive generation.
            kw = {}
            if self.cfg.steer == "restrict":
                toks = self._prefixes.tokens(policy.pid)
                if toks:
                    kw["allowed_first"] = toks
            elif self.cfg.steer == "bias":
                w = self._prefixes.weights(policy.pid)
                if w:
                    kw.update(bias=w, bias_steps=self.cfg.steer_steps,
                              bias_scale=self.cfg.steer_scale)
            elif self.cfg.steer == "hidden" and self._centroids is not None:
                v = self._centroids.get(policy.pid)
                if v is not None:
                    kw.update(hidden_vec=v, hidden_steps=self.cfg.steer_steps,
                              hidden_alpha=self.cfg.steer_alpha)
            out = steered_generate(self.model, self.tokenizer, prompt_ids,
                                   self.gen.with_(max_new_tokens=max_new),
                                   device=self.device, **kw)
            self._calls += 1
            reply = self.tokenizer.decode(out).strip()
            tr.candidates = [reply]
        elif self.cfg.output_controls and policy is not None:
            cands = self._candidates(prompt_ids, max_new, self.cfg.n_candidates)
            tr.candidates = cands
            # Consistency-based selection is only applied when the policy is
            # TRUSTWORTHY. Measured: same-model exemplar scoring classifies the
            # policy at 18-19% accuracy, and steering generation with a wrong
            # policy is far worse than not steering at all -- it took Core-Bench
            # from 0.675 to 0.458. Deterministic shortcuts and oracle policies
            # are trusted; a model-inferred policy is used only for length.
            # The learned head is trusted: 92.2% held-out accuracy against 19.2%
            # for same-model exemplar scoring. Trust is granted on measured
            # accuracy, not on the mechanism being newer.
            # Two levels of trust, separated because they were measured to
            # behave differently. Length and question constraints are safe under
            # an imperfect policy -- a wrong length is a mild error. Exemplar
            # CONSISTENCY matching is not: it actively pulls generation toward
            # the wrong act, and at 76.9% classification accuracy that cost more
            # than the 23% of errors were worth. Consistency is therefore
            # reserved for policies that are correct by construction.
            reply = self._rerank(cands, policy,
                                 use_consistency=source in ("shortcut", "oracle"))
        else:
            out = generate(self.model, self.tokenizer, prompt_ids,
                           self.gen.with_(max_new_tokens=max_new), device=self.device)
            self._calls += 1
            reply = self.tokenizer.decode(out).strip()
        tr.generation = reply

        # --- validation ---------------------------------------------------
        recent = [h["content"] for h in self.history if h["role"] == "assistant"]
        if self.cfg.validator:
            n_tok = len(self.tokenizer.encode(reply))
            verdict = validate(reply, policy=policy, n_tokens=n_tok, recent=recent,
                               closing=self.state.conversation_closing)
            tr.validator = verdict.as_dict()
            if not verdict.ok and self.cfg.output_controls and policy is not None:
                # Exactly one controlled retry: re-rank the remaining candidates
                # rather than regenerate, which would cost another k passes for a
                # model that has already told us its preferences.
                alts = [c for c in tr.candidates if c != reply]
                if alts:
                    reply = self._rerank(alts, policy)
                    tr.retried = True
                    n_tok = len(self.tokenizer.encode(reply))
                    tr.validator = validate(reply, policy=policy, n_tokens=n_tok,
                                            recent=recent,
                                            closing=self.state.conversation_closing
                                            ).as_dict()

        if not reply:
            reply = "..."
        self.history.append({"role": "assistant", "content": reply})
        self.state.observe_reply(reply, policy.pid if policy else None,
                                 len(self.tokenizer.encode(reply)))

        tr.final = reply
        tr.state = self.state.as_dict()
        tr.model_calls = self._calls
        tr.latency_ms = (time.perf_counter() - t0) * 1000
        return reply
