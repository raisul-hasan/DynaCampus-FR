class DecisionState:
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"

class OpenSetDecisionEngine:
    """
    4-State Open-Set Decision Engine with Adaptive Margin (AOSM).
    """
    def __init__(self, base_threshold=0.45, quality_threshold=0.40):
        self.base_threshold = base_threshold
        self.quality_threshold = quality_threshold

    def evaluate(self, top_matches, probe_quality):
        """
        Evaluate probe against top gallery matches.
        top_matches: list of tuples (identity_id, similarity_score, metadata)
        probe_quality: float (0.0 to 1.0)
        
        Returns: (DecisionState, BestMatchID, BestScore)
        """
        if probe_quality < self.quality_threshold:
            return DecisionState.INSUFFICIENT_QUALITY, None, 0.0
            
        if not top_matches:
            return DecisionState.UNKNOWN, None, 0.0

        best_id, best_score, best_meta = top_matches[0]

        # Adaptive Threshold (AOSM novelty)
        # If quality is perfectly 1.0, threshold is strict (base_threshold)
        # If quality is around quality_threshold (0.40), threshold relaxes slightly
        # Penalty = (1.0 - quality) * 0.1, so max relaxation is -0.06
        penalty = (1.0 - probe_quality) * 0.1
        adaptive_threshold = max(self.base_threshold - penalty, 0.25)
        
        # Ambiguous Zone Margin (0.05 below adaptive threshold)
        ambiguous_margin = adaptive_threshold - 0.05

        if best_score >= adaptive_threshold:
            return DecisionState.KNOWN, best_id, best_score
        elif best_score >= ambiguous_margin:
            return DecisionState.AMBIGUOUS, best_id, best_score
        else:
            return DecisionState.UNKNOWN, None, best_score
