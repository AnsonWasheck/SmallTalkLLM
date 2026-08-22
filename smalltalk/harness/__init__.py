"""A small deterministic conversational operating layer around SmallTalkLLM.

The base model is never modified. Everything here is external orchestration:
deterministic bookkeeping, explicit state, bounded memory, constrained policy
selection, and output validation. The research question is how much apparent
conversational competence a 6,689,024-parameter model gains when it is relieved
of the work ordinary software does exactly.
"""

from .config import HarnessConfig, MODES          # noqa: F401
from .runner import Harness                        # noqa: F401
