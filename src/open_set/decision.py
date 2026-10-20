import math


class DecisionState:
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"

class OpenSetDecisionEngine:
    """Four-state rule using calibrated similarity, top-two margin and quality.

    Thresholds are fixed at inference. Quality never relaxes the match threshold.
    A missing second score denotes a gallery containing just one candidate.
    """
    def __init__(self, base_threshold=0.45, quality_threshold=0.40, margin_threshold=0.05):
        if not math.isfinite(base_threshold) or not -1.000001 <= base_threshold <= 1.000001:
            raise ValueError('Similarity threshold must be finite and within the cosine range.')
        if not math.isfinite(quality_threshold) or not 0 <= quality_threshold <= 1:
            raise ValueError('Quality threshold must be in [0, 1].')
        if not math.isfinite(margin_threshold) or not 0 <= margin_threshold <= 2:
            raise ValueError('Margin threshold must be in [0, 2].')
        self.base_threshold = base_threshold
        self.quality_threshold = quality_threshold
        self.margin_threshold = margin_threshold

    def evaluate(self, top_matches, probe_quality):
        """
        Evaluate probe against top gallery matches.
        top_matches: list of tuples (identity_id, similarity_score, metadata)
        probe_quality: float (0.0 to 1.0)
        
        Returns: (DecisionState, BestMatchID, BestScore)
        """
        if not top_matches:
            return self.evaluate_scores(None, 0.0, None, probe_quality)
        second_score = float(top_matches[1][1]) if len(top_matches) > 1 else None
        return self.evaluate_scores(top_matches[0][0], float(top_matches[0][1]), second_score, probe_quality)

    def evaluate_scores(self, top1_id, top1_score, top2_score, probe_quality):
        """Return (state, accepted_identity_or_None, top1_score).

        Apply quality -> minimum score -> top1-minus-top2 margin -> known.
        Callers must pass both scores whenever their gallery contains two or more IDs.
        """
        if not math.isfinite(probe_quality) or not 0 <= probe_quality <= 1:
            raise ValueError('Probe quality must be finite and in [0, 1].')
        if not math.isfinite(top1_score):
            raise ValueError('Top-one score must be finite.')
        if top2_score is not None and (not math.isfinite(top2_score) or top2_score > top1_score):
            raise ValueError('Top-two scores must be finite and ranked.')
        if probe_quality < self.quality_threshold:
            return DecisionState.INSUFFICIENT_QUALITY, None, top1_score
        if top1_id is None or top1_score < self.base_threshold:
            return DecisionState.UNKNOWN, None, top1_score
        if top2_score is not None and top1_score - top2_score < self.margin_threshold:
            return DecisionState.AMBIGUOUS, None, top1_score
        return DecisionState.KNOWN, top1_id, top1_score
