# pyrefly: ignore [missing-import]
import pytest
from src.open_set.decision import OpenSetDecisionEngine, DecisionState

def test_insufficient_quality_decision():
    osde = OpenSetDecisionEngine(base_threshold=0.45, quality_threshold=0.40)
    decision, match_id, score = osde.evaluate([("S001", 0.90, None)], probe_quality=0.20)
    assert decision == DecisionState.INSUFFICIENT_QUALITY
    assert match_id is None

def test_known_decision():
    osde = OpenSetDecisionEngine(base_threshold=0.45, quality_threshold=0.40)
    decision, match_id, score = osde.evaluate([("S001", 0.85, None)], probe_quality=0.80)
    assert decision == DecisionState.KNOWN
    assert match_id == "S001"

def test_unknown_decision():
    osde = OpenSetDecisionEngine(base_threshold=0.45, quality_threshold=0.40)
    decision, match_id, score = osde.evaluate([("S001", 0.10, None)], probe_quality=0.80)
    assert decision == DecisionState.UNKNOWN
