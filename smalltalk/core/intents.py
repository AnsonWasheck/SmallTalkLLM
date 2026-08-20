"""SmallTalk Core: the fundamental conversational states.

Design rule, and it is the whole point of this module:

    BROAD input paraphrases, NARROW output behaviour.

Everything before v0.2 optimised for diversity, which spread probability mass
across a dozen equally-plausible replies and made greedy decoding a coin flip.
Here we deliberately drive H(Y|X) down: each intent has one canonical target
(occasionally two), and a slightly wider `accept` set used only for scoring so
the evaluator does not punish a reply that is genuinely right.

`train` paraphrases and `held_out` paraphrases are disjoint by construction --
held_out is never emitted by the generator, only by the evaluator. That is what
makes a Core-100 score a generalisation measurement rather than a lookup test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CORE_VERSION = "core-v0.2.1"   # +out_of_scope

# response-length policy tokens; single source of truth is the tokenizer, which
# reserves them inside the 4096-entry budget.
from ..tokenizer import LENGTH_TOKENS as LEN_TOKENS  # noqa: E402

_NORM = re.compile(r"[^a-z0-9' ]+")


def normalise(text: str) -> str:
    t = _NORM.sub(" ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Intent:
    name: str
    length: str                       # reaction | vshort | short | medium
    targets: list[str]                # what we TRAIN on (narrow, usually 1)
    train: list[str]                  # input paraphrases seen in training
    held_out: list[str]               # input paraphrases used ONLY for eval
    accept: list[str] = field(default_factory=list)   # scored-correct replies
    tier: int = 1                     # 1 = trivial, must hit 99%; 3 = harder

    def accepted(self) -> set[str]:
        return {normalise(t) for t in (self.targets + self.accept)}

    def is_correct(self, reply: str) -> bool:
        return normalise(reply) in self.accepted()


def _i(name, length, targets, train, held_out, accept=(), tier=1) -> Intent:
    return Intent(name, length, list(targets), list(train), list(held_out),
                  list(accept), tier)


INTENTS: list[Intent] = [
    _i("greeting", "vshort", ["hey"],
       ["hi", "hello", "hey", "yo", "hey there", "hiya", "heya", "hello there",
        "hi!", "hey!", "morning", "good morning", "evening", "sup", "hey you",
        "oi", "hello?", "hi hi", "hey :)", "hello!", "hey there!", "hi again",
        "yo!", "ey up", "alright", "hey stranger"],
       ["howdy", "hey hey", "good evening", "hullo", "yo yo", "hi there!",
        "afternoon", "greetings"],
       accept=["hi", "hello", "hey there", "hey!", "hi!"]),

    _i("greeting_how_are_you", "short", ["i'm good, how about you?"],
       ["hello how are you", "hi how are you", "hey how are you",
        "hi, how are you?", "hey how're you doing", "hello how's it going",
        "yo how you doing", "hi there how have you been", "hey, you good?",
        "hey how are things", "hi how's things", "hello, how are you doing?",
        "hey how you been", "hi, you alright?", "hey there how's it going",
        "morning, how are you"],
       ["howdy, how are you", "hey how's your day going", "hi how have you been",
        "hello, all good?", "hey you doing ok", "yo, how's things",
        "good morning how are you", "hi there, you well?"],
       accept=["i'm good, you?", "good, how about you?", "i'm alright, you?",
               "good thanks, how about you?"]),

    _i("how_are_you", "short", ["not bad, you?"],
       ["how are you", "how're you", "how are you doing", "how's it going",
        "how you doing", "you good?", "you alright?", "how's things",
        "how have you been", "how's your day", "how's life", "you ok?",
        "what's up", "how goes it", "everything good?"],
       ["how are things with you", "how've you been", "you doing ok?",
        "how's your day been", "what's new", "all good?", "how's everything",
        "you well?"],
       accept=["not bad, how about you?", "i'm good, you?", "pretty good, you?",
               "not bad. you?"]),

    _i("thanks", "reaction", ["no worries"],
       ["thanks", "thank you", "cheers", "thanks a lot", "thanks!", "ty",
        "thank you so much", "appreciate it", "thanks for that",
        "thanks, that helps", "cheers mate", "much appreciated"],
       ["thank you kindly", "thanks a bunch", "big thanks", "ta",
        "appreciate that", "thanks so much", "cheers for that"],
       accept=["no problem", "anytime", "you're welcome", "np", "of course"]),

    _i("goodbye", "vshort", ["see you"],
       ["bye", "goodbye", "see ya", "later", "gtg", "i'm off", "night",
        "goodnight", "catch you later", "i have to go", "bye!", "talk later",
        "heading off", "see you later", "i'm gonna head off", "right, i'm going",
        "ok i'm off now", "time for bed", "signing off", "gotta go",
        "i'll leave you to it", "speak soon"],
       ["cya", "i'm heading out", "gotta run", "night night",
        "off to bed", "i should go", "take care"],
       accept=["see ya", "bye", "later", "see you later", "take care", "night"]),

    _i("apology", "short", ["it's alright"],
       ["sorry", "i'm sorry", "my bad", "sorry about that", "apologies",
        "sorry!", "that was my fault", "sorry, my mistake", "oops sorry",
        "i apologise"],
       ["sorry for that", "my apologies", "that's on me", "sorry, my bad",
        "i messed up sorry"],
       accept=["no worries", "it's fine", "all good", "don't worry about it",
               "no problem"]),

    _i("good_news", "short", ["that's great!"],
       ["i got the job", "i passed my exam", "we won", "i got promoted",
        "she said yes", "i finally finished it", "got some good news today",
        "i got in!", "my test came back clear", "i got an offer",
        "we're moving in together", "i sold the car"],
       ["i got accepted", "i aced it", "we got the house", "i got the callback",
        "the results were good", "i hit my target", "i graduated"],
       accept=["that's amazing", "congrats!", "that's great", "nice one",
               "that's brilliant", "congratulations"], tier=2),

    _i("bad_news", "short", ["oh no, i'm sorry"],
       ["i lost my job", "my dog died", "we broke up", "i failed the exam",
        "my grandma passed away", "i didn't get it", "things went badly",
        "i got rejected", "my car got written off", "she's in hospital",
        "i got made redundant"],
       ["i lost my grandad", "we split up", "i didn't get the job",
        "my cat passed", "it fell through", "i got some bad news",
        "the results were bad"],
       accept=["oh no, sorry", "i'm sorry", "that's awful, i'm sorry",
               "oh no. i'm sorry", "sorry to hear that"], tier=2),

    _i("bored", "short", ["same, nothing going on here"],
       ["i'm bored", "so bored", "nothing to do", "i'm so bored right now",
        "bored out of my mind", "there's nothing on", "i'm bored lol",
        "dead boring today", "nothing happening", "nothing on today",
        "just sat here doing nothing", "got nothing on", "quiet one today",
        "bored", "nothing much happening"],
       ["bored as anything", "unbelievably bored", "i've got nothing to do",
        "boring day", "nothing going on"],
       accept=["same here", "same", "yeah it's a slow one",
               "same, nothing here either"], tier=2),

    _i("tired", "short", ["long day?"],
       ["i'm tired", "so tired", "i'm exhausted", "i'm knackered",
        "i'm shattered", "didn't sleep", "i'm wiped", "running on empty",
        "so sleepy", "i'm drained", "i need sleep", "shattered", "so tired",
        "could sleep for a week", "i'm running on fumes", "knackered"],
       ["absolutely exhausted", "i'm beat", "barely slept", "worn out",
        "i'm so tired today"],
       accept=["rough day?", "long one?", "get some rest", "you should sleep"],
       tier=2),

    _i("confused", "short", ["what do you mean?"],
       ["i don't get it", "what?", "huh?", "i'm confused", "that makes no sense",
        "sorry i don't follow", "come again?", "wait what", "i don't understand",
        "sorry what?", "you lost me", "i'm not following", "run that by me again",
        "what was that", "i missed that"],
       ["eh?", "i'm lost", "what do you mean", "not following", "you what"],
       accept=["what do you mean", "sorry, what?", "which bit?"], tier=2),

    _i("agreement", "reaction", ["yeah exactly"],
       ["i agree", "exactly", "yeah true", "you're right", "same", "definitely",
        "100%", "couldn't agree more", "yep", "for sure", "yeah same",
        "that's true", "you're not wrong", "i think so too", "right?",
        "yeah i reckon", "makes sense"],
       ["totally", "agreed", "true that", "spot on", "yeah i think so too"],
       accept=["yeah", "right?", "exactly", "same here"], tier=2),

    _i("disagreement", "short", ["really? i don't see it"],
       ["i disagree", "nah", "i don't think so", "not really", "i'm not sure about that",
        "hmm i don't agree", "that's not right", "nope"],
       ["i don't buy it", "disagree", "not convinced", "i doubt it"],
       accept=["really?", "hm, i'm not sure", "you reckon?"], tier=3),

    _i("user_vents", "short", ["that sounds rough"],
       ["work has been a nightmare", "everyone's been on my case",
        "i've had it up to here", "i'm so done with this week",
        "nothing is going right", "i can't deal with them anymore",
        "it's been one thing after another"],
       ["i'm at my limit", "this week has been brutal",
        "everything keeps going wrong", "i'm so fed up"],
       accept=["that sounds hard", "ugh, that's rough", "sounds exhausting"],
       tier=2),

    _i("user_jokes", "reaction", ["haha"],
       ["lol", "haha", "that's so funny", "i'm dying", "lmao", "hahaha",
        "that cracked me up", "😂"],
       ["lololol", "ha", "that's hilarious", "i can't stop laughing"],
       accept=["ha", "hahaha", "lol", "that's good"], tier=2),

    _i("user_asks_opinion", "short", ["what do you think?"],
       ["what do you reckon", "what do you think", "thoughts?", "your opinion?",
        "would you do it", "should i?", "what would you do"],
       ["any thoughts", "what's your take", "do you think i should",
        "reckon it's worth it"],
       accept=["depends, what's your gut say?", "hard to say", "i'd probably go for it"],
       tier=3),

    _i("greeting_plus_name", "short", ["hey, how's it going?"],
       ["hey it's me", "hi it's sam", "hello it's me again", "hey, me again",
        "hi, it's your mate"],
       ["hey it's alex", "hello, me again", "hi it's me!"],
       accept=["hey!", "hey, how are you?", "hey you"], tier=2),

    _i("check_in", "short", ["yeah still here"],
       ["you there?", "you still there?", "still around?", "you alive?",
        "anyone there", "you awake?"],
       ["still there?", "you around?", "hey you there"],
       accept=["yep, here", "still here", "yeah i'm here"], tier=2),

    _i("compliment", "short", ["ah thanks"],
       ["you're great", "you're the best", "i like talking to you",
        "you're funny", "thanks for listening", "you always know what to say"],
       ["you're lovely", "i enjoy this", "you're good at this"],
       accept=["thanks!", "that's kind", "ah cheers"], tier=2),

    _i("small_plan", "short", ["sounds good"],
       ["fancy a coffee?", "want to meet up?", "shall we do saturday?",
        "you free later?", "pub tonight?", "want to grab food?"],
       ["fancy a walk?", "up for lunch?", "you around this weekend?"],
       accept=["yeah sounds good", "sure", "yeah i'm up for that", "sounds fun"],
       tier=2),

    # The "none of the above" class. Without it the model has 20 buckets and no
    # way to decline, so it answers factual questions and nonsense with whichever
    # social reflex is nearest -- measured: "what is the square root of nine" and
    # "my hovercraft is full of eels" both returned "that sounds rough". A Core
    # score without this class is flattering, because it never tests the case
    # where the right answer is "not one of my twenty phrases".
    #
    # Note this deliberately does NOT cover ordinary off-topic chat: that is
    # handled by the real-dialogue portion of the SFT mix, which supplies genuine
    # human replies. This class covers input the model should refuse to
    # pattern-match: factual queries, tasks, and noise.
    _i("out_of_scope", "short", ["no idea honestly"],
       ["what is the capital of france", "what's 17 times 4",
        "who invented the telephone", "write me a poem",
        "translate this to spanish", "what year did the war end",
        "summarise this article", "how do i fix a segfault",
        "asdfgh", "zxcvbn qwerty", "define entropy",
        "what's the square root of nine", "give me a python function",
        "how many miles to the moon", "explain quantum tunnelling",
        "convert 40c to fahrenheit", "qqqqqqqq", "spell onomatopoeia"],
       ["what is the population of brazil", "what's 23 plus 19",
        "who wrote hamlet", "write me an essay", "how do i install docker",
        "what's the boiling point of water", "hjkl asdf", "define recursion",
        "what's 8 squared"],
       accept=["no idea", "not sure", "dunno", "no clue", "couldn't tell you",
               "no idea, sorry", "beats me"], tier=3),
]

BY_NAME = {i.name: i for i in INTENTS}

assert len(BY_NAME) == len(INTENTS), "duplicate intent name"
_seen_surface: dict[str, str] = {}
for _it in INTENTS:
    for _p in _it.train + _it.held_out:
        _n = normalise(_p)
        _owner = _seen_surface.setdefault(_n, _it.name)
        assert _owner == _it.name, f"{_n!r} claimed by both {_owner} and {_it.name}"
    _overlap = {normalise(x) for x in _it.train} & {normalise(x) for x in _it.held_out}
    assert not _overlap, f"{_it.name}: held-out leakage {_overlap}"
