"""Writer and critic prompts for the Qwen3.5-9B upstream teacher.

Design notes:

* The writer receives ONE latent spec and realises it. It never invents structure.
* The spec is described in prose in the *system* turn; the dialogue must contain no
  trace of the metadata (no "family:", no "[callback]", no stage directions).
* Anti-assistant-speak guidance is phrased as *distributional* ("must not dominate"),
  not as a hard phrase ban — a mechanical ban produces stilted text and the brief
  explicitly warns against it.
* The critic is a SEPARATE invocation with no access to the writer's context, and runs
  at low temperature. It rejects rather than repairs.
"""

from __future__ import annotations

import json
from typing import Any

WRITER_VERSION = "writer-v1.0.0"
CRITIC_VERSION = "critic-v1.0.0"

WRITER_SYSTEM = """\
You write realistic text-message conversations between two ordinary people.

You will be given a hidden SPEC describing the situation. Realise it as natural dialogue.
Output ONLY JSON. No markdown fences, no commentary, no analysis, no <think>.

FORMAT (exactly this shape):
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Rules:
- Roles strictly alternate, starting with "user" and ending with "assistant".
- "assistant" is a FRIEND texting back, NOT an AI, NOT a helper, NOT a therapist.
  Never mention being an AI, an assistant, or a model. No meta-commentary.
- Write how people actually text: lowercase drift, contractions, fragments, the
  occasional typo, trailing thoughts, backchannels ("yeah", "mm", "lol", "idk",
  "wait what", "fair", "no way", "ohh"). These repeating is fine and correct.
- Reply length must follow the situation, NOT a fixed target. Mix very short reactions
  ("lol", "oof", "same"), normal short replies, medium replies, and the occasional
  longer one when the moment actually calls for it.
- The assistant does not interrogate. Not every turn ends in a question.
- Advice only when a friend would actually give it. No lists. No summaries.
- These patterns may appear occasionally but must NOT dominate the conversation:
  "I'm sorry to hear that", "Would you like to talk about it?", "I'm here for you",
  "That sounds really hard", "I'd be happy to help". Vary reactions instead.
- NEVER put the spec's labels, plan steps, or metadata into the dialogue text.
- Follow the spec's factual constraints EXACTLY: if it says a fact was never stated,
  the user must never state it; if a fact is corrected, the correction must happen.
"""


def _describe_spec(spec: dict[str, Any]) -> str:
    """Render the latent spec as prose instructions for the writer."""
    e = spec.get("entities", {})
    lines: list[str] = []
    lines.append(f"Relationship: the two are {spec['relationship']}.")
    lines.append(f"Setting: {spec['setting']}.")
    lines.append(f"The user is feeling: {spec['user_mood']}.")
    lines.append(f"The friend replying writes like: {spec['assistant_register']}.")
    lines.append(f"Main topic: {spec['topic']}.")
    lines.append(f"What the user wants from the chat: {spec['conversation_goal']}.")
    lines.append(f"Total messages: about {spec['target_turns']} "
                 f"(a {spec['length_profile']} conversation).")

    if e.get("people"):
        who = ", ".join(f"{p['name']} (their {p['relation']})" for p in e["people"])
        lines.append(f"People who may be mentioned by name: {who}.")
    if e.get("pet"):
        lines.append(f"They have a {e['pet']['kind']} called {e['pet']['name']}.")
    if e.get("job"):
        lines.append(f"The user works as a {e['job']}.")
    if e.get("hobby"):
        lines.append(f"The user is into {e['hobby']}.")
    if e.get("place"):
        lines.append(f"Somewhere relevant: {e['place']}.")

    if spec.get("known_facts"):
        lines.append("Must happen: " + "; ".join(spec["known_facts"]) + ".")
    if spec.get("unknown_facts"):
        lines.append("IMPORTANT constraint: " + "; ".join(spec["unknown_facts"]) + ".")
    if spec.get("state_mutations"):
        lines.append("A fact must change mid-conversation: "
                     + "; ".join(spec["state_mutations"]) + ".")
    if spec.get("contradictions"):
        lines.append("Contradiction to include: " + "; ".join(spec["contradictions"]) + ".")
    if spec.get("callbacks"):
        lines.append("Callback requirement: " + "; ".join(spec["callbacks"]) + ".")
    if spec.get("sarcasm"):
        lines.append("Include sarcasm: the user says something positive-sounding that "
                     "clearly means the opposite; the friend reads it correctly.")
    if spec.get("ambiguity"):
        lines.append("Include a genuine misunderstanding that gets cleared up naturally.")
    if spec.get("reference_distance"):
        lines.append(f"The recalled detail should sit roughly {spec['reference_distance']} "
                     f"tokens before the question — put real conversation in between, "
                     f"not filler.")

    lines.append("Rough shape of the conversation (do NOT label these in the text): "
                 + " -> ".join(spec["discourse_plan"]) + ".")
    return "\n".join("- " + l for l in lines)


def writer_messages(spec: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content":
            "SPEC:\n" + _describe_spec(spec) +
            "\n\nWrite the conversation now. JSON only."},
    ]


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
CRITIC_SYSTEM = """\
You are a strict quality reviewer for text-message dialogue data.

You receive REQUIREMENTS and a CONVERSATION. Judge the conversation only.
Output ONLY JSON. No commentary, no markdown fences, no <think>.

FORMAT (exactly this shape):
{"scores":{"natural":0-5,"register":0-5,"coherence":0-5,"reference_correct":0-5,
"state_correct":0-5,"emotion_fit":0-5,"non_assistant":0-5,"non_repetitive":0-5,
"length_fit":0-5},"violations":["..."],"verdict":"accept"|"reject"}

Scoring guidance:
- natural: reads like two real people texting, not generated prose.
- register: casual friend voice; NOT customer service, NOT therapy, NOT an AI helper.
- coherence: every reply actually responds to the message before it.
- reference_correct: pronouns/names/details point at the right things.
- state_correct: if a fact was corrected, later mentions use the NEW value; if a fact
  was never stated, the friend does NOT invent it.
- emotion_fit: reaction matches what happened (no cheerfulness at bad news).
- non_assistant: free of "I'm here for you"/"Would you like to talk about it" DOMINANCE,
  no lists, no unprompted advice-giving, no AI self-reference.
- non_repetitive: replies aren't near-copies of each other or of the user's words.
- length_fit: reply lengths vary and suit the moment.

Reject if ANY of: any score <= 2; the assistant invents a fact never stated; a required
correction did not happen; the assistant reveals it is an AI; metadata/labels leak into
the text; roles do not alternate; the text contains reasoning or <think>.
Be strict. It is better to reject a mediocre sample than to accept it.
"""


def critic_messages(spec: dict[str, Any], conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    req: list[str] = [f"- Register expected: {spec['assistant_register']}",
                      f"- User mood: {spec['user_mood']}",
                      f"- Topic: {spec['topic']}"]
    if spec.get("known_facts"):
        req.append("- Must contain: " + "; ".join(spec["known_facts"]))
    if spec.get("unknown_facts"):
        req.append("- MUST NOT invent: " + "; ".join(spec["unknown_facts"]))
    if spec.get("state_mutations"):
        req.append("- Fact must change and the LATEST value must be used later: "
                   + "; ".join(spec["state_mutations"]))
    if spec.get("contradictions"):
        req.append("- Contradiction handling: " + "; ".join(spec["contradictions"]))
    if spec.get("callbacks"):
        req.append("- Callback required: " + "; ".join(spec["callbacks"]))
    if spec.get("sarcasm"):
        req.append("- Sarcasm must be present and correctly read")
    if spec.get("ambiguity"):
        req.append("- A misunderstanding must occur and be resolved")

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content":
            "REQUIREMENTS:\n" + "\n".join(req) +
            "\n\nCONVERSATION:\n" + convo +
            "\n\nReview now. JSON only."},
    ]
