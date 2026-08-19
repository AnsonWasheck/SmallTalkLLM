"""Synthetic conversational data: teacher prompt specs + an offline fallback generator.

Two paths, one schema:

1. TEACHER PATH (preferred for the real study). `build_teacher_prompts()` emits
   generation requests for a larger model; the returned JSONL is ingested by
   `adapters.load_jsonl_conversations`. No teacher is needed at inference time.

2. OFFLINE PATH. `generate_offline_corpus()` composes conversations from a
   template grammar. It is *not* a substitute for teacher data -- it exists so the
   repo is end-to-end runnable, smoke tests are hermetic, and the pipeline can be
   validated before spending money. Its limited diversity is a known confound and
   is reported as such in docs/REPORT.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .schema import Conversation, Turn

TOPICS = [
    "greeting", "checking_in", "how_was_your_day", "work", "school", "food",
    "hobbies", "music", "games", "movies", "sleep", "exercise", "friends",
    "family", "weather", "plans", "boredom", "excitement", "frustration",
    "disappointment", "mild_sadness", "celebration", "jokes", "teasing",
    "topic_change", "clarification", "repair", "goodbye", "unknown_fact",
]

EMOTIONS = ["neutral", "tired", "happy", "excited", "bored", "annoyed",
            "sad", "stressed", "relieved", "amused", "curious"]

TEACHER_SYSTEM_PROMPT = """\
You generate training data for a very small casual-conversation model.

Write a natural one-on-one text conversation between two friends: `user` and `assistant`.

Hard rules:
- The assistant is a FRIEND, not an assistant. Never helpful, never formal.
- Assistant replies: 3-25 words, one or two sentences. Often much shorter.
- Lowercase-leaning, contractions, mild slang, occasional "lol"/"oof"/"honestly".
- React to feelings first. Ask at most one follow-up question, and not every turn.
- No lists, no advice-giving, no disclaimers, no "as an AI", no factual lecturing.
- If the user asks something factual the assistant wouldn't know, it says so
  casually ("honestly no idea lol") and redirects to the person. Never invent facts.
- Occasional light humour and playful teasing. Never mean.
- Vary openings. Do not start consecutive replies with the same word.

Output STRICT JSON only:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
"""

CANDIDATE_SYSTEM_PROMPT = """\
Given a casual conversation ending on a user turn, write N alternative assistant
replies. Same rules: friend not assistant, 3-25 words, no advice, no facts invented.
Make the candidates genuinely different in strategy (empathise / joke / ask / react).

Output STRICT JSON only:
{"candidates": ["...", "...", "..."]}
"""


@dataclass
class TeacherRequest:
    id: str
    topic: str
    emotion: str
    scenario: str
    num_turns: int
    style: str = "casual texting, 3-25 word assistant replies"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "system_prompt": TEACHER_SYSTEM_PROMPT,
            "topic": self.topic,
            "emotion": self.emotion,
            "scenario": self.scenario,
            "num_turns": self.num_turns,
            "style": self.style,
            "user_prompt": (
                f"Topic: {self.topic}. Emotion: {self.emotion}. "
                f"Situation: {self.scenario}. Write {self.num_turns} total turns "
                f"(alternating, starting with user)."
            ),
        }


SCENARIO_BANK: dict[str, list[str]] = {
    "work": ["a day of back-to-back meetings", "an annoying coworker", "finishing a big project",
             "a boring shift", "getting praised by their boss"],
    "school": ["cramming for an exam", "a group project falling apart", "a class they love",
               "getting a grade back"],
    "food": ["burning dinner", "trying a new restaurant", "being out of groceries",
             "craving something specific late at night"],
    "sleep": ["not sleeping well for days", "oversleeping and missing something",
              "weird dreams"],
    "music": ["a new album they love", "a concert next month", "learning an instrument badly"],
    "games": ["being stuck on a boss", "replaying an old favourite", "a friend beating them"],
    "movies": ["a movie that disappointed them", "bingeing a series", "wanting a recommendation"],
    "exercise": ["starting to run again", "sore from the gym", "skipping the gym guiltily"],
    "friends": ["a friend moving away", "a hangout that got cancelled", "an inside joke"],
    "family": ["a sibling visiting", "a call with their mom", "a family dinner"],
    "weather": ["endless rain", "the first warm day", "unexpected snow"],
    "plans": ["a trip next month", "an empty weekend", "cancelling plans to stay in"],
    "boredom": ["nothing to do on a sunday", "scrolling for an hour", "waiting for something"],
    "excitement": ["good news they can't hold in", "a thing they've waited months for"],
    "frustration": ["tech that won't work", "being stuck in traffic", "a plan falling through"],
    "disappointment": ["not getting something they wanted", "a letdown of an evening"],
    "mild_sadness": ["feeling lonely in a new city", "missing someone", "a flat grey week"],
    "celebration": ["a new job", "a birthday", "finishing something hard"],
    "jokes": ["a dumb thing that happened to them", "a pun they're proud of"],
    "teasing": ["being bad at cooking", "their terrible music taste"],
    "topic_change": ["switching from work to food mid-conversation"],
    "clarification": ["being vague about 'the thing' and needing to explain"],
    "repair": ["mishearing each other and sorting it out"],
    "goodbye": ["having to leave suddenly", "winding down for the night"],
    "greeting": ["just saying hi", "checking in after a few days"],
    "checking_in": ["asking how the other's week went"],
    "how_was_your_day": ["an ordinary uneventful day"],
    "hobbies": ["a hobby they just picked up", "not having time for a hobby"],
    "unknown_fact": ["asking an obscure trivia question mid-chat"],
}


def build_teacher_prompts(
    n: int = 5000, seed: int = 0, turn_range: tuple[int, int] = (6, 16)
) -> list[TeacherRequest]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        scenario = rng.choice(SCENARIO_BANK.get(topic, ["an ordinary day"]))
        reqs.append(
            TeacherRequest(
                id=f"syn-{i:06d}",
                topic=topic,
                emotion=rng.choice(EMOTIONS),
                scenario=scenario,
                num_turns=rng.randrange(turn_range[0], turn_range[1] + 1, 2),
            )
        )
    return reqs


def write_teacher_prompts(path: str | Path, requests: Sequence[TeacherRequest]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# Offline template generator
# ---------------------------------------------------------------------------
OPENERS_USER = ["hey", "yo", "hi", "hey you", "heyy", "sup", "hey there", "morning",
                "evening", "hey, you around?"]
OPENERS_ASST = ["hey", "hey, what's up?", "yo", "hey you", "hi! how's it going?",
                "hey, how are you?", "sup", "oh hey", "hey :) what's new?"]

ACK_NEG = ["oof.", "damn.", "ugh, that sucks.", "that's rough.", "aw man.",
           "yikes.", "oh no.", "well that's not great.", "hm, sorry."]
ACK_POS = ["oh nice!", "yesss.", "that's awesome.", "no way, congrats!", "love that.",
           "hey that's great.", "amazing.", "let's go."]
ACK_NEU = ["yeah?", "mm.", "oh?", "gotcha.", "fair.", "makes sense.", "right.",
           "huh.", "for real."]

FOLLOWUPS = ["what happened?", "how come?", "long day?", "you doing ok?", "since when?",
             "how'd that go?", "what'd you do?", "any better now?", "tell me more?",
             "so what now?", "you ok?", "was it bad?"]
REACTS = ["that's the worst kind of tired lol.", "honestly same.", "i'd be annoyed too.",
          "at least it's over.", "you earned a break.", "sounds about right.",
          "classic.", "that tracks.", "big mood.", "hope it gets easier."]
JOKES = ["you chose violence, i respect it.", "the smoke alarm is just a loud applause.",
         "sounds like a you problem lol.", "bold strategy.", "10/10 chaos.",
         "you're a menace lol."]
UNKNOWN = ["honestly no idea lol.", "no clue tbh.", "beats me.", "never heard of it.",
           "not a clue, what is it?", "hmm, dunno."]
GOODBYE_USER = ["ok i gotta go", "alright, later", "im heading out", "gotta run",
                "ok talk tomorrow", "night", "bye!"]
GOODBYE_ASST = ["later!", "night, sleep well.", "take care :)", "see ya.",
                "byee.", "talk soon.", "have a good one."]

USER_LINES: dict[str, list[tuple[str, str]]] = {
    # (user utterance, valence)
    "work": [("work was brutal today", "neg"), ("mostly meetings", "neg"),
             ("my boss piled on more stuff", "neg"), ("i actually finished the project", "pos"),
             ("shift dragged forever", "neg"), ("got some good feedback today", "pos"),
             ("i have to work saturday", "neg"), ("my coworker quit today", "neu"),
             ("deadline got pushed back thank god", "pos"),
             ("six meetings back to back", "neg"), ("i got nothing done today", "neg"),
             ("my inbox is a disaster", "neg"), ("wfh today at least", "pos")],
    "school": [("i have an exam friday", "neg"), ("group project is a mess", "neg"),
               ("i got an A actually", "pos"), ("class was kinda fun today", "pos"),
               ("i havent started the essay", "neg"), ("finals are next week", "neg"),
               ("my professor is so boring", "neg"), ("i passed!", "pos"),
               ("pulled an all nighter studying", "neg"),
               ("nobody in my group does anything", "neg")],
    "food": [("i burned dinner lol", "neu"), ("im so hungry", "neu"),
             ("tried that new place downtown", "pos"), ("theres nothing in the fridge", "neg"),
             ("made pasta from scratch", "pos"), ("i ate way too much", "neu"),
             ("craving pizza so bad", "neu"), ("coffee is the only thing keeping me alive", "neu"),
             ("forgot to eat lunch again", "neg"), ("the tacos were incredible", "pos")],
    "sleep": [("i barely slept", "neg"), ("kinda tired today", "neg"),
              ("i slept like 10 hours", "pos"), ("had the weirdest dream", "neu"),
              ("up til 3am for no reason", "neg"), ("i need a nap so bad", "neg"),
              ("slept through my alarm", "neg"), ("actually rested for once", "pos"),
              ("my neighbors were loud all night", "neg")],
    "music": [("found a new album i love", "pos"), ("im learning guitar badly", "neu"),
              ("theres a show next month", "pos"), ("this song is stuck in my head", "neu"),
              ("saw a band live last night", "pos"), ("my playlist is all sad songs lately", "neu"),
              ("i cant stop replaying one track", "pos")],
    "games": [("im stuck on this boss", "neg"), ("replaying an old rpg", "pos"),
              ("my friend destroyed me lol", "neu"), ("finally beat it!!", "pos"),
              ("i rage quit honestly", "neg"), ("been playing way too much lately", "neu"),
              ("the new update broke everything", "neg")],
    "movies": [("that movie was a letdown", "neg"), ("bingeing a show right now", "pos"),
               ("need something to watch", "neu"), ("the ending made no sense", "neg"),
               ("i cried at the end ngl", "neu"), ("watched three episodes in a row", "pos"),
               ("everyone says its good but idk", "neu")],
    "exercise": [("started running again", "pos"), ("im so sore", "neu"),
                 ("skipped the gym again", "neg"), ("hit a new pr today", "pos"),
                 ("my legs are dead", "neu"), ("went for a long walk", "pos"),
                 ("i keep saying ill go tomorrow", "neg")],
    "friends": [("my friend is moving away", "neg"), ("plans got cancelled", "neg"),
                ("hung out with sam yesterday", "pos"), ("we talked for like four hours", "pos"),
                ("i havent seen anyone in weeks", "neg"), ("my roommate is driving me nuts", "neg"),
                ("everyone bailed last minute", "neg")],
    "family": [("my sister is visiting", "pos"), ("called my mom for an hour", "neu"),
               ("family dinner was chaos", "neu"), ("my dad is teaching me to drive", "neu"),
               ("my cousin had a baby", "pos"), ("my brother borrowed my car again", "neg")],
    "weather": [("its been raining all week", "neg"), ("finally warm out", "pos"),
                ("its freezing here", "neg"), ("it snowed overnight", "neu"),
                ("perfect day out today", "pos"), ("so humid i cant breathe", "neg")],
    "plans": [("nothing planned this weekend", "neu"), ("im going away next month", "pos"),
              ("thinking of staying in tonight", "neu"), ("we might do dinner friday", "pos"),
              ("i should probably make plans", "neu"), ("cancelled everything to rest", "pos")],
    "boredom": [("im so bored", "neu"), ("nothing to do", "neu"), ("idk", "neu"),
                ("ive been scrolling for an hour", "neu"), ("nothing sounds fun rn", "neu"),
                ("just waiting around", "neu"), ("someone entertain me lol", "neu")],
    "excitement": [("guess what", "pos"), ("i got the job!!", "pos"), ("i cant wait", "pos"),
                   ("i have news", "pos"), ("its finally happening", "pos"),
                   ("im so hyped right now", "pos"), ("you'll never believe this", "pos"),
                   ("i start monday", "pos")],
    "frustration": [("my laptop keeps crashing", "neg"), ("stuck in traffic forever", "neg"),
                    ("nothing is working today", "neg"), ("i lost all my work", "neg"),
                    ("been on hold for 40 minutes", "neg"), ("my phone died at the worst time", "neg")],
    "mild_sadness": [("been feeling kinda lonely", "neg"), ("i miss my old place", "neg"),
                     ("just a flat kind of week", "neg"), ("i dont really know anyone here", "neg"),
                     ("everything feels quiet lately", "neg"), ("i miss my friends", "neg")],
    "celebration": [("its my birthday today", "pos"), ("we finished it finally", "pos"),
                    ("i graduated!!", "pos"), ("we're celebrating tonight", "pos"),
                    ("everything actually worked out", "pos")],
    "jokes": [("i tried to cook today", "neu"), ("i walked into a door lol", "neu"),
              ("i called my teacher mom", "neu"), ("i wore two different shoes to work", "neu"),
              ("i waved at someone who wasnt waving at me", "neu")],
    "unknown_fact": [("who invented the spinning jenny", "neu"),
                     ("whats the tallest building in peru", "neu"),
                     ("do you know what year the fax machine came out", "neu"),
                     ("whats the capital of kyrgyzstan", "neu"),
                     ("how do submarines actually work", "neu")],
}

# Topic-conditioned follow-ups. Without these, replies are drawn from one global
# pool and a "guess what" can be answered with a tiredness quip -- which is exactly
# the incoherence we are trying to measure, so it must not be baked into the data.
TOPIC_FOLLOWUPS: dict[str, list[str]] = {
    "work": ["how many meetings?", "is it always like that?", "when do you get a break?",
             "your boss again?", "did you at least eat?", "long day huh?"],
    "school": ["what class?", "when's it due?", "did you study at all?",
               "are you ready for it?", "how'd the rest go?"],
    "food": ["what'd you make?", "was it any good?", "did you order instead?",
             "what are you craving?", "did you save me any?"],
    "sleep": ["how many hours?", "couldn't switch off?", "nap later?",
              "what kept you up?", "any better tonight?"],
    "music": ["who is it?", "what genre?", "are you going?", "send it to me?",
              "on repeat all day?"],
    "games": ["which one?", "how long you been stuck?", "did you beat it?",
              "worth playing?", "how many tries?"],
    "movies": ["what'd you watch?", "would you recommend it?", "how many episodes deep?",
               "was the ending bad?", "what genre you feeling?"],
    "exercise": ["how far?", "first time back?", "sore tomorrow?", "gym or outside?"],
    "friends": ["when do they leave?", "how long has it been?", "did they say why?",
                "you two close?", "gonna reschedule?"],
    "family": ["how long are they staying?", "how's she doing?", "everyone good?",
               "were they nice about it?"],
    "weather": ["still raining?", "warm enough to go out?", "how cold?",
                "did it stick?", "getting out today?"],
    "plans": ["anything you wanna do?", "who's going?", "when?",
              "you feeling up for it?", "where to?"],
    "boredom": ["what do you usually do?", "wanna pick something?", "nothing at all?",
                "seen anything good lately?", "what would help?"],
    "excitement": ["what what??", "tell me!", "since when?", "how'd it happen?",
                   "when do you start?", "no way, really?"],
    "frustration": ["what happened?", "how long has it been doing that?",
                    "did you try restarting it?", "still stuck?", "that's the worst."],
    "mild_sadness": ["how long has it felt like that?", "you doing ok?",
                     "wanna talk about it?", "anyone you can see this week?",
                     "that sounds heavy."],
    "celebration": ["congrats!!", "how are you celebrating?", "who's coming?",
                    "you must be thrilled.", "that's huge!"],
    "jokes": ["please tell me someone saw", "and then what", "amazing.",
              "you're a menace lol", "did anyone notice?"],
    "unknown_fact": ["no idea, what made you think of it?", "why do you ask?",
                     "where'd you see that?", "is it important?"],
}

TOPIC_REACTS: dict[str, list[str]] = {
    "work": ["meetings about the work you could be doing.", "that's a whole day gone.",
             "you've earned a quiet evening.", "hope tomorrow's lighter."],
    "school": ["you'll get through it.", "future you will be grateful.",
               "group projects are a scam honestly."],
    "food": ["food fixes a lot honestly.", "cooking is chaos, respect.",
             "now im hungry too."],
    "sleep": ["that's the worst kind of tired lol.", "sleep debt is real.",
              "go to bed early, seriously."],
    "music": ["good music carries a week.", "i love finding something new.",
              "that's a whole mood."],
    "games": ["that boss sounds personal now.", "nostalgia hits different.",
              "one more try energy."],
    "movies": ["endings are so hard to land.", "that's a solid way to spend a night.",
               "the hype is usually a lie."],
    "exercise": ["starting again is the hard part.", "sore means it counted.",
                 "tomorrow you'll feel it."],
    "friends": ["distance is rough.", "that kind of talk is the best kind.",
                "people are flaky lately."],
    "family": ["family is a lot sometimes.", "that's sweet honestly.",
               "hope it's a good visit."],
    "weather": ["this week has been relentless.", "finally, some relief.",
                "perfect excuse to stay in."],
    "plans": ["an empty weekend is underrated.", "that sounds fun.",
              "resting counts as plans."],
    "boredom": ["boredom spiral incoming.", "dangerous energy.",
                "go make a snack, trust me."],
    "excitement": ["that's amazing!", "ok im excited for you now.",
                   "you deserve this.", "let's gooo."],
    "frustration": ["technology is fake.", "id be losing it too.",
                    "that would ruin my whole day."],
    "mild_sadness": ["that sounds really lonely.", "im glad you said something.",
                     "im here if you wanna talk."],
    "celebration": ["so happy for you!", "that's worth celebrating properly.",
                    "huge. genuinely."],
    "jokes": ["you chose violence, i respect it.", "10/10 chaos.", "bold strategy."],
    "unknown_fact": ["my brain has no entry for that lol.", "we can look it up together.",
                     "wild that you'd wonder that."],
}

FILLER_USER = ["yeah", "kinda", "i guess", "mhm", "for real", "exactly", "true",
               "yeah pretty much", "eh", "not really", "sort of", "honestly yeah"]


@dataclass
class OfflineConfig:
    num_conversations: int = 4000
    turn_range: tuple[int, int] = (6, 16)
    seed: int = 0
    candidate_fraction: float = 0.0   # >0 emits stage-3 candidate records
    candidates_per_example: int = 4
    weights: dict[str, float] = field(default_factory=dict)


def generate_offline_corpus(cfg: OfflineConfig | None = None) -> Iterator[Conversation]:
    cfg = cfg or OfflineConfig()
    rng = random.Random(cfg.seed)
    topics = [t for t in USER_LINES]

    for i in range(cfg.num_conversations):
        topic = topics[i % len(topics)]
        n_pairs = rng.randrange(cfg.turn_range[0], cfg.turn_range[1] + 1) // 2
        msgs: list[Turn] = []
        pool = list(USER_LINES[topic])
        rng.shuffle(pool)

        msgs.append(Turn("user", rng.choice(OPENERS_USER)))
        msgs.append(Turn("assistant", rng.choice(OPENERS_ASST)))

        for k in range(1, n_pairs):
            if pool and rng.random() < 0.7:
                utt, valence = pool.pop()
            else:
                utt, valence = rng.choice(FILLER_USER), "neu"
            if k == n_pairs - 1 and rng.random() < 0.6:
                utt, valence = rng.choice(GOODBYE_USER), "bye"
            msgs.append(Turn("user", utt))
            msgs.append(Turn("assistant", _reply(rng, topic, valence, k)))

        conv = Conversation(
            id=f"offline-{i:06d}",
            messages=msgs,
            source="offline_template",
            meta={"topic": topic, "generator": "template_v1"},
        )
        if cfg.candidate_fraction and rng.random() < cfg.candidate_fraction:
            last_valence = "neu"
            from .schema import Candidate

            conv.candidates = [
                Candidate(content=_candidate(rng, topic, last_valence), source="template_teacher")
                for _ in range(cfg.candidates_per_example)
            ]
        yield conv


def _reply(rng: random.Random, topic: str, valence: str, turn: int) -> str:
    """Compose a reply that is conditioned on the topic, not just the valence.

    Topic-specific follow-ups/reactions are strongly preferred over the global
    pools so the corpus never teaches "any reply fits any turn".
    """
    if valence == "bye":
        return rng.choice(GOODBYE_ASST)
    if topic == "unknown_fact" and rng.random() < 0.75:
        # graceful ignorance, optionally plus a redirect back to the person
        out = rng.choice(UNKNOWN)
        if rng.random() < 0.5:
            out += " " + rng.choice(TOPIC_FOLLOWUPS["unknown_fact"])
        return out
    if topic == "jokes" and rng.random() < 0.45:
        return rng.choice(JOKES)

    ack = {"neg": ACK_NEG, "pos": ACK_POS}.get(valence, ACK_NEU)
    followups = TOPIC_FOLLOWUPS.get(topic, FOLLOWUPS)
    reacts = TOPIC_REACTS.get(topic, REACTS)

    parts = []
    if rng.random() < 0.65:
        parts.append(rng.choice(ack))
    r = rng.random()
    if r < 0.50:
        parts.append(rng.choice(followups if rng.random() < 0.85 else FOLLOWUPS))
    elif r < 0.85:
        parts.append(rng.choice(reacts if rng.random() < 0.85 else REACTS))
    if not parts:
        parts.append(rng.choice(followups))
    return " ".join(parts)


def _candidate(rng: random.Random, topic: str, valence: str) -> str:
    return _reply(rng, topic, valence, 1)


def write_offline_corpus(path: str | Path, cfg: OfflineConfig | None = None) -> int:
    from .schema import write_jsonl

    return write_jsonl(path, generate_offline_corpus(cfg))
