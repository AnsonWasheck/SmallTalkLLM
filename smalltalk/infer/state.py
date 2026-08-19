"""Optional zero-parameter application-level conversation state.

This adds *no* model parameters and no learned components: it is regex/keyword
bookkeeping in the application layer, injected as a short system hint. It exists so
we can answer "how much of apparent memory can be bought outside the weights?"
It is OFF by default -- pure-model performance must remain separately measurable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "work": ("work", "job", "boss", "meeting", "shift", "office", "deadline", "coworker"),
    "school": ("school", "class", "exam", "homework", "teacher", "assignment", "college"),
    "food": ("food", "eat", "dinner", "lunch", "breakfast", "hungry", "cook", "pizza", "coffee"),
    "sleep": ("sleep", "tired", "nap", "bed", "insomnia", "exhausted", "awake"),
    "music": ("music", "song", "album", "band", "playlist", "concert", "guitar"),
    "games": ("game", "gaming", "playstation", "xbox", "steam", "level", "match"),
    "movies": ("movie", "film", "show", "series", "netflix", "episode", "watched"),
    "exercise": ("gym", "run", "workout", "walk", "lift", "yoga", "swim"),
    "friends": ("friend", "buddy", "hang out", "party", "roommate"),
    "family": ("mom", "dad", "sister", "brother", "family", "parents", "cousin"),
    "weather": ("weather", "rain", "snow", "sunny", "cold", "hot", "storm"),
    "plans": ("weekend", "tomorrow", "tonight", "plan", "trip", "vacation", "later"),
    "mood": ("sad", "happy", "excited", "bored", "stressed", "anxious", "annoyed", "great"),
}

# Casual texters rarely capitalise, so match either case and reject the words
# that legitimately follow "i'm" (moods, states) rather than requiring a capital.
NAME_RE = re.compile(
    r"\b(?:i'?m|i am|my name(?:'s| is)|call me)\s+([A-Za-z]{2,15})\b", re.IGNORECASE
)
NOT_A_NAME = {
    "tired", "sorry", "good", "fine", "ok", "okay", "great", "sad", "happy",
    "bored", "busy", "back", "here", "home", "late", "done", "hungry", "so",
    "just", "not", "really", "still", "kinda", "pretty", "very", "excited",
    "stressed", "exhausted", "annoyed", "sick", "free", "off", "out", "in",
    "at", "on", "a", "an", "the", "gonna", "going", "trying", "thinking",
    "glad", "worried", "lonely", "confused", "curious", "new", "old", "fed",
}
PET_RE = re.compile(r"\bmy (dog|cat|pet)(?:'s name is|,? )?\s*([A-Z][a-z]{1,15})?", re.IGNORECASE)
FACT_RE = re.compile(
    r"\bi (?:have|got|just|really|always|never|hate|love|like|started|finished|bought)\b[^.!?]{0,60}",
    re.IGNORECASE,
)
MOOD_WORDS = {
    "tired": ("tired", "exhausted", "sleepy", "drained", "wiped"),
    "sad": ("sad", "down", "bummed", "upset", "lonely"),
    "stressed": ("stressed", "anxious", "overwhelmed", "swamped", "brutal"),
    "happy": ("happy", "great", "awesome", "amazing", "stoked", "excited"),
    "bored": ("bored", "boring", "nothing to do", "meh"),
    "annoyed": ("annoyed", "angry", "mad", "frustrated", "pissed"),
}


@dataclass
class ConversationState:
    current_topic: str | None = None
    topic_history: list[str] = field(default_factory=list)
    user_name: str | None = None
    user_mood: str | None = None
    user_details: list[str] = field(default_factory=list)
    max_details: int = 5

    def observe(self, role: str, text: str) -> None:
        if role != "user":
            return
        low = text.lower()

        topic = self.detect_topic(low)
        if topic and topic != self.current_topic:
            self.current_topic = topic
            self.topic_history.append(topic)

        for mood, words in MOOD_WORDS.items():
            if any(w in low for w in words):
                self.user_mood = mood
                break

        m = NAME_RE.search(text)
        if m and m.group(1).lower() not in NOT_A_NAME:
            self.user_name = m.group(1).capitalize()

        for f in FACT_RE.findall(text):
            detail = " ".join(f.split())
            if detail and detail not in self.user_details:
                self.user_details.append(detail)
        del self.user_details[: max(0, len(self.user_details) - self.max_details)]

    @staticmethod
    def detect_topic(low: str) -> str | None:
        best, score = None, 0
        for topic, words in TOPIC_KEYWORDS.items():
            hits = sum(w in low for w in words)
            if hits > score:
                best, score = topic, hits
        return best

    def as_system_hint(self) -> str:
        """Short, tokenizer-cheap hint. Kept under ~25 tokens on purpose."""
        parts = []
        if self.user_name:
            parts.append(f"name {self.user_name}")
        if self.current_topic:
            parts.append(f"topic {self.current_topic}")
        if self.user_mood:
            parts.append(f"mood {self.user_mood}")
        if self.user_details:
            parts.append(self.user_details[-1])
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "current_topic": self.current_topic,
            "topic_history": self.topic_history,
            "user_name": self.user_name,
            "user_mood": self.user_mood,
            "user_details": self.user_details,
        }
