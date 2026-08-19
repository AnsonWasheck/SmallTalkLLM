# smalltalk-ai v2 — teacher corpus generation spec

Run this against a frontier model and return a JSONL bundle. The 6,689,024-parameter
student never sees the latent specification — only the resulting natural conversation.

**Target: 30–50M unique tokens. Uniqueness matters more than volume.** Our existing
synthetic sources failed on exactly this axis: `social_gold` has an 89% utterance-repeat
rate, my template generator 70–97%. A 6.7M model saturates a low-entropy distribution
almost immediately — validation loss sat at 0.205 for five straight rounds while
behaviour still visibly changed.

---

## 1. Output schema (one JSON object per line)

```json
{
  "id": "t2-000001",
  "family": "grief_disclosure_v1",
  "split_hint": "train",
  "latent": {
    "relationship": "close friend, ~2 years",
    "register": "lowercase texter, dry humour",
    "user_emotion": "numb, underplaying it",
    "topic": "bereavement",
    "dialogue_acts": ["greeting","disclosure","support","boundary","close"],
    "implicit_information": "user does not want advice, wants presence",
    "facts_established": {"grandmother_name": "Ada", "funeral": "friday"},
    "facts_corrected": {},
    "ambiguity": null,
    "sarcasm": false,
    "uncertainty": null
  },
  "messages": [
    {"role": "user", "content": "hey"},
    {"role": "assistant", "content": "hey you. how's today been?"}
  ]
}
```

**`family` is mandatory and load-bearing.** It names the generating grammar/scenario
template. We split train/val/test **by family**, never by example. Aim for **≥300
distinct families**, no family exceeding 2% of the corpus.

`split_hint` is advisory; we re-split by family on ingest.

---

## 2. Hard style constraints (these are the product)

- Assistant is a **friend, not an assistant**. Never helpful-by-default, never formal.
- Assistant replies **3–25 words**, one or two sentences. Often shorter.
- Lowercase-leaning, contractions, natural disfluency. Vary sentence openings —
  do not start consecutive replies with the same word.
- **No advice unless explicitly asked.** No "you should", no numbered lists, no
  disclaimers, no "as an AI", no therapy-speak.
- Never invent facts the user did not supply.
- At most one question per reply, and not every turn.
- Conversations **6–24 turns**; at least 25% of the corpus ≥16 turns
  (we train at 1024 context and currently have almost no genuinely long dialogue).

---

## 3. Latent taxonomy — sample independently per dialogue

| axis | values |
|---|---|
| relationship | new acquaintance · close friend · sibling · partner · coworker · old friend reconnecting |
| register | lowercase texter · punctuation-heavy · emoji-light · dry · warm · terse · rambly |
| user_emotion | neutral · tired · elated · anxious · grieving · bored · irritated · wistful · giddy · flat |
| topic | 40+ everyday domains (work, food, sleep, pets, travel, hobbies, weather, family, money worries, health niggles, plans, media) |
| dialogue_acts | greeting · disclosure · probe · reassurance · teasing · anecdote · clarification · repair · topic-drift · callback · boundary · close |
| implicit_information | what the user means but does not say |
| facts_established | key→value facts the user states |
| facts_corrected | facts later changed (drives memory-update skill) |
| ambiguity | vague referent the assistant should query |
| sarcasm | surface/intent mismatch |
| uncertainty | something the assistant cannot know |

---

## 4. Required skill mix

Our measured failures, in priority order. Percentages are of dialogues.

| skill | share | what it must teach |
|---|---|---|
| **epistemic discrimination** | 12% | see §5 — the single weakest area (0.50–0.62) |
| **memory / state mutation** | 12% | see §6 |
| multi-turn coherence w/ plan | 15% | topic → disclosure → clarification → anecdote → drift → **callback** |
| implicit emotion | 8% | feeling implied, never named |
| valence-correct support | 8% | grief/bad news never met with cheer |
| boundary respect | 6% | user deflects; assistant does not push |
| sarcasm / irony | 6% | surface positive, intent negative |
| repair & correction | 6% | user corrects a misunderstanding |
| humour & teasing | 6% | play along, never explain the joke |
| low-signal input | 6% | "yeah" / "mm" / "idk" without degenerating |
| celebration | 5% | positive news, proportionate warmth |
| ordinary chat | 10% | the connective tissue |

---

## 5. Epistemic discrimination (12%) — paired, not one-sided

Do **not** simply add more "I don't know". That trains an indiscriminate refusal
reflex. Every item must be a **contrastive pair over the same context**:

| condition | correct behaviour |
|---|---|
| context contains the answer | answer confidently |
| context does not contain it | say you don't know |
| user stated it earlier | recall it |
| user never stated it | do not invent it |
| fact was corrected later | use the latest version |
| question is ambiguous | acknowledge the ambiguity |

For each, also emit a **plausible hallucinated negative** for DPO:

```json
{"prompt":[...], "chosen":"you never told me his name",
 "rejected":"i think you said his name was jake", "tag":"fabricated_recall"}
```

Target ~20–50k such pairs across the six conditions.

---

## 6. Memory / state mutation (12%)

Build dialogues around **state changes across distance**, at full length:

```
turn 2 : my sister is Maya
turn 7 : Maya hates hiking
turn 11: actually she's really into hiking now
turn 17: what does my sister think about hiking?
```

Vary: distance between statement and query (4–18 turns), number of facts (1–4),
number of corrections (0–2), and whether the queried fact was ever stated
(negative cases are essential — they teach *not* inventing).

---

## 7. Conversation plans (15%)

Generate from an **invisible plan**, not by independently sampling plausible replies:

```
topic A → disclosure → clarification → related anecdote → minor drift → callback to A → close
```

Emit the plan into `latent.dialogue_acts`; never into the visible conversation.

---

## 8. Preference pairs (separate file)

`preferences.jsonl`, same schema as your `social_gold_preferences.jsonl`. Most
valuable negatives are **our student's actual failure modes**, not generic bad text:

```json
{"prompt":[{"role":"user","content":"yeah idk today was just weird"}],
 "chosen":"weird how?",
 "rejected":"I'm sorry to hear that you're having a weird day. Would you like to talk about what happened?",
 "rejection_tag":"assistant_verbosity"}
```

Tags to cover: `assistant_verbosity`, `advice_unprompted`, `fabricated_recall`,
`valence_mismatch`, `ignored_boundary`, `explained_the_joke`, `generic_filler`,
`repetition`, `context_copy`.

---

## 9. Deliverable

```
teacher_v2_bundle/
  conversations.jsonl      # 30-50M unique tokens, family-tagged
  preferences.jsonl        # 20-50k pairs
  families.json            # family -> {count, skill, description}
  stats.json               # token count, per-skill/per-family counts,
                           # exact_utterance_repeat_fraction  <- report this
```

**Acceptance gates (I verify all of these on ingest, and will report failures):**

1. `exact_utterance_repeat_fraction` **< 0.25** (ours were 0.89 and 0.97)
2. ≥300 families, none >2% of corpus
3. Assistant reply mean 8–14 words, ≥90% within 3–25
4. ≥25% of dialogues ≥16 turns
5. **Zero 6-gram overlap with SmallTalkBench-v2** (I enforce this; contaminated
   families are dropped, not tolerated)
6. <1% of assistant turns matching advice/assistant-verbosity patterns

---

## 10. What happens on our side

```
ingest → clean → leakage gate → split BY FAMILY (train/val/test)
      → dense causal pretraining (all tokens, 1024 ctx)
      → A/B: random-init vs warm-start from real7m
      → assistant-masked SFT
      → DPO on preference pairs
      → int4 export → 3.88 MB
```

Architecture stays frozen at **6,689,024 parameters** throughout.
