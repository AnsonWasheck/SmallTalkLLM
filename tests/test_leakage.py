from smalltalk.data.schema import Conversation, Turn
from smalltalk.eval.bench_v2 import build_scenarios
from smalltalk.eval.leakage import check_conversations


def test_v2_benchmark_overlap_fails_closed():
    # Exact six-token span from an immutable v2 scenario, selected from the
    # generated frozen set rather than duplicating a surface string here.
    words = build_scenarios()[0].user_turns[1].split()
    text = " ".join(words[:6])
    c = Conversation(
        id="leak", source="test",
        messages=[Turn("user", text),
                  Turn("assistant", "nice")],
    )
    report = check_conversations([c])
    assert not report.clean
    assert report.flagged
