# -*- coding: utf-8 -*-
"""
Researcher Progressive Enablement - Phase 4 Optimization

Advanced optimization with machine learning-style decision making,
dynamic threshold adjustment, and intelligent auto-tuning capabilities.
"""

import logging
import time
import statistics
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import Phase 3 components
from .researcher_progressive_enablement import (
    ScenarioContext,
    ResearcherProgressiveEnabler as BaseEnabler,
)

logger = logging.getLogger(__name__)


@dataclass
class AdvancedScenarioFeatures:
    """Extended scenario features for advanced analysis"""

    base_scenario: ScenarioContext

    # Temporal features
    time_of_day: int = field(
        default_factory=lambda: int(time.time() % 86400)
    )  # Seconds since midnight
    day_of_week: int = field(default_factory=lambda: int(time.time() // 86400) % 7)

    # Context features
    context_growth_rate: float = 0.0  # How fast context is growing
    query_complexity_score: float = 0.0  # Computed complexity of queries
    data_diversity_score: float = 0.0  # How diverse the data sources are

    # Performance features
    recent_performance_trend: float = 0.0  # Recent performance trend
    resource_pressure: float = 0.0  # Current system resource pressure

    # User behavior features
    user_interaction_frequency: float = 0.0  # How often user interacts
    task_urgency_score: float = 0.5  # Estimated task urgency (0-1)


@dataclass
class DecisionConfidence:
    """Confidence metrics for isolation decisions"""

    confidence_score: float  # 0-1, how confident we are in the decision
    contributing_factors: List[str]  # What factors led to this decision
    uncertainty_sources: List[str]  # What makes us uncertain
    alternative_probability: float  # Probability that opposite decision would be better


@dataclass
class DynamicThreshold:
    """Dynamic threshold that adjusts based on performance"""

    name: str
    current_value: float
    min_value: float
    max_value: float
    adjustment_rate: float = 0.1
    performance_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def adjust(self, performance_feedback: float):
        """Adjust threshold based on performance feedback"""
        self.performance_history.append(performance_feedback)

        if len(self.performance_history) < 5:
            return  # Need more data

        # Calculate recent trend
        recent_avg = statistics.mean(list(self.performance_history)[-10:])
        older_avg = (
            statistics.mean(list(self.performance_history)[-20:-10:])
            if len(self.performance_history) >= 20
            else recent_avg
        )

        trend = recent_avg - older_avg

        # Adjust threshold based on trend
        if trend > 0.1:  # Performance improving
            # Lower threshold to be more aggressive
            adjustment = -self.adjustment_rate * abs(trend)
        elif trend < -0.1:  # Performance degrading
            # Raise threshold to be more conservative
            adjustment = self.adjustment_rate * abs(trend)
        else:
            adjustment = 0

        self.current_value = max(
            self.min_value, min(self.max_value, self.current_value + adjustment)
        )
        logger.debug(
            f"Adjusted threshold {self.name}: {self.current_value:.2f} (trend: {trend:.3f})"
        )


class AdvancedResearcherProgressiveEnabler(BaseEnabler):
    """Advanced progressive enablement with machine learning-style optimization"""

    def __init__(self):
        super().__init__()

        # Advanced learning parameters
        self.feature_weights: Dict[str, float] = {
            "step_count": 1.0,
            "context_size": 1.0,
            "parallel_execution": 0.8,
            "token_estimate": 1.2,
            "complexity_score": 1.5,
            "time_of_day": 0.3,
            "resource_pressure": 0.7,
            "user_urgency": 0.9,
        }

        # Dynamic thresholds
        self.dynamic_thresholds = {
            "step_count": DynamicThreshold("step_count", 3.0, 1.0, 8.0),
            "context_size": DynamicThreshold("context_size", 5000.0, 1000.0, 15000.0),
            "token_estimate": DynamicThreshold(
                "token_estimate", 10000.0, 3000.0, 30000.0
            ),
            "complexity_score": DynamicThreshold("complexity_score", 0.7, 0.3, 0.9),
        }

        # Advanced tracking
        self.decision_history: deque = deque(maxlen=1000)
        self.feature_importance: Dict[str, float] = defaultdict(float)
        self.pattern_cache: Dict[str, Tuple[bool, float, float]] = (
            {}
        )  # pattern -> (decision, confidence, timestamp)

        # Auto-tuning parameters
        self.auto_tune_enabled = True
        self.tune_frequency = 50  # Tune every N decisions
        self.decision_count = 0

        # Performance prediction
        self.performance_predictor = PerformancePredictor()

        logger.info(
            "Initialized AdvancedResearcherProgressiveEnabler with ML-style optimization"
        )

    def extract_advanced_features(
        self, scenario: ScenarioContext
    ) -> AdvancedScenarioFeatures:
        """Extract advanced features from scenario"""
        features = AdvancedScenarioFeatures(base_scenario=scenario)

        # Calculate query complexity score
        description = scenario.task_description.lower()
        complexity_keywords = {
            "analyze": 2,
            "compare": 2,
            "synthesize": 3,
            "evaluate": 2,
            "comprehensive": 2,
            "detailed": 1,
            "thorough": 2,
            "in-depth": 3,
            "multiple": 1,
            "various": 1,
            "different": 1,
            "cross-reference": 3,
        }

        features.query_complexity_score = (
            sum(
                weight
                for keyword, weight in complexity_keywords.items()
                if keyword in description
            )
            / 10.0
        )  # Normalize

        # Calculate context growth rate (estimate)
        if scenario.step_count > 1:
            features.context_growth_rate = scenario.context_size / scenario.step_count

        # Estimate resource pressure based on current metrics
        from .researcher_isolation_metrics import get_isolation_metrics

        metrics = get_isolation_metrics()
        health = metrics.get_isolation_health()

        # Simple resource pressure estimation
        recent_sessions = health.get("recent_sessions_count", 0)
        features.resource_pressure = min(
            1.0, recent_sessions / 10.0
        )  # Normalize to 0-1

        # Estimate task urgency based on parallel execution and step count
        if scenario.parallel_execution and scenario.step_count >= 3:
            features.task_urgency_score = 0.8
        elif scenario.step_count >= 5:
            features.task_urgency_score = 0.7
        else:
            features.task_urgency_score = 0.5

        return features

    def calculate_isolation_score(self, features: AdvancedScenarioFeatures) -> float:
        """Calculate isolation score using weighted features"""
        scenario = features.base_scenario
        score = 0.0

        # Base features with dynamic thresholds
        if scenario.step_count >= self.dynamic_thresholds["step_count"].current_value:
            score += self.feature_weights["step_count"]

        if (
            scenario.context_size
            >= self.dynamic_thresholds["context_size"].current_value
        ):
            score += self.feature_weights["context_size"]

        if (
            scenario.estimated_tokens
            >= self.dynamic_thresholds["token_estimate"].current_value
        ):
            score += self.feature_weights["token_estimate"]

        if (
            features.query_complexity_score
            >= self.dynamic_thresholds["complexity_score"].current_value
        ):
            score += self.feature_weights["complexity_score"]

        # Boolean features
        if scenario.parallel_execution:
            score += self.feature_weights["parallel_execution"]

        # Advanced features
        score += features.resource_pressure * self.feature_weights["resource_pressure"]
        score += features.task_urgency_score * self.feature_weights["user_urgency"]

        # Time-based adjustment
        hour = features.time_of_day // 3600
        if 9 <= hour <= 17:  # Business hours - more conservative
            score *= 0.9
        elif 22 <= hour or hour <= 6:  # Night hours - more aggressive
            score *= 1.1

        return score

    def predict_performance_impact(
        self, features: AdvancedScenarioFeatures, enable_isolation: bool
    ) -> Tuple[float, float]:
        """Predict performance impact of isolation decision"""
        return self.performance_predictor.predict(features, enable_isolation)

    def should_enable_isolation_advanced(
        self, scenario: ScenarioContext
    ) -> Tuple[bool, str, Dict[str, Any], DecisionConfidence]:
        """Advanced isolation decision with confidence metrics"""
        features = self.extract_advanced_features(scenario)

        # Check for cached patterns
        pattern_key = self._generate_pattern_key(features)
        if pattern_key in self.pattern_cache:
            cached_decision, cached_confidence, timestamp = self.pattern_cache[
                pattern_key
            ]
            if time.time() - timestamp < 3600:  # Cache valid for 1 hour
                return (
                    cached_decision,
                    "Cached decision from similar pattern",
                    {"pattern_key": pattern_key},
                    DecisionConfidence(
                        confidence_score=cached_confidence,
                        contributing_factors=["pattern_matching"],
                        uncertainty_sources=[],
                        alternative_probability=1 - cached_confidence,
                    ),
                )

        # Calculate isolation score
        isolation_score = self.calculate_isolation_score(features)

        # Predict performance impact
        perf_with_isolation, perf_without_isolation = self.predict_performance_impact(
            features, True
        )

        # Calculate confidence
        confidence = self._calculate_decision_confidence(features, isolation_score)

        # Make decision
        decision_threshold = 3.0  # Base threshold

        # Adjust threshold based on confidence
        if confidence.confidence_score < 0.6:
            decision_threshold *= 1.2  # Be more conservative when uncertain
        elif confidence.confidence_score > 0.9:
            decision_threshold *= 0.8  # Be more aggressive when confident

        enable_isolation = isolation_score >= decision_threshold

        # Generate explanation
        reason = self._generate_decision_explanation(
            features, isolation_score, decision_threshold, confidence
        )

        # Prepare decision factors
        decision_factors = {
            "isolation_score": isolation_score,
            "decision_threshold": decision_threshold,
            "predicted_perf_with_isolation": perf_with_isolation,
            "predicted_perf_without_isolation": perf_without_isolation,
            "feature_contributions": self._get_feature_contributions(features),
            "dynamic_thresholds": {
                k: v.current_value for k, v in self.dynamic_thresholds.items()
            },
        }

        # Cache decision
        self.pattern_cache[pattern_key] = (
            enable_isolation,
            confidence.confidence_score,
            time.time(),
        )

        # Record decision for learning
        self._record_decision(features, enable_isolation, isolation_score, confidence)

        # Auto-tune if needed
        self.decision_count += 1
        if self.auto_tune_enabled and self.decision_count % self.tune_frequency == 0:
            self._auto_tune_parameters()

        return enable_isolation, reason, decision_factors, confidence

    def should_enable_isolation_with_explanation(
        self, scenario: ScenarioContext
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Wrapper method for backward compatibility with tests"""
        enable, reason, factors, confidence = self.should_enable_isolation_advanced(
            scenario
        )
        factors["confidence"] = confidence.confidence_score
        return enable, reason, factors

    def _generate_pattern_key(self, features: AdvancedScenarioFeatures) -> str:
        """Generate a pattern key for caching similar scenarios"""
        scenario = features.base_scenario

        # Discretize continuous features for pattern matching
        step_bucket = (scenario.step_count // 2) * 2
        context_bucket = (scenario.context_size // 2000) * 2000
        token_bucket = (scenario.estimated_tokens // 5000) * 5000
        complexity_bucket = int(features.query_complexity_score * 10) / 10

        return f"{step_bucket}_{context_bucket}_{token_bucket}_{complexity_bucket}_{scenario.parallel_execution}_{scenario.has_search_results}"

    def _calculate_decision_confidence(
        self, features: AdvancedScenarioFeatures, isolation_score: float
    ) -> DecisionConfidence:
        """Calculate confidence in the isolation decision"""
        contributing_factors = []
        uncertainty_sources = []

        # Base confidence from isolation score clarity
        score_clarity = abs(isolation_score - 3.0)  # Distance from threshold
        base_confidence = min(1.0, score_clarity / 2.0)

        # Adjust based on historical data availability
        similar_scenarios = self._find_similar_scenarios(features.base_scenario)
        if len(similar_scenarios) >= 5:
            contributing_factors.append("sufficient_historical_data")
            base_confidence *= 1.2
        else:
            uncertainty_sources.append("limited_historical_data")
            base_confidence *= 0.8

        # Adjust based on feature consistency
        scenario = features.base_scenario
        if (
            scenario.step_count >= 3
            and scenario.context_size >= 5000
            and scenario.estimated_tokens >= 10000
        ):
            contributing_factors.append("consistent_complexity_indicators")
            base_confidence *= 1.1
        elif (
            scenario.step_count <= 1
            and scenario.context_size <= 1000
            and scenario.estimated_tokens <= 3000
        ):
            contributing_factors.append("consistent_simplicity_indicators")
            base_confidence *= 1.1
        else:
            uncertainty_sources.append("mixed_complexity_signals")
            base_confidence *= 0.9

        # Adjust based on resource pressure
        if features.resource_pressure > 0.8:
            uncertainty_sources.append("high_resource_pressure")
            base_confidence *= 0.9

        confidence_score = max(0.1, min(1.0, base_confidence))
        alternative_probability = 1.0 - confidence_score

        return DecisionConfidence(
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            uncertainty_sources=uncertainty_sources,
            alternative_probability=alternative_probability,
        )

    def _generate_decision_explanation(
        self,
        features: AdvancedScenarioFeatures,
        isolation_score: float,
        threshold: float,
        confidence: DecisionConfidence,
    ) -> str:
        """Generate human-readable explanation for the decision"""
        scenario = features.base_scenario

        if isolation_score >= threshold:
            explanation = f"Enable isolation (score: {isolation_score:.1f} >= threshold: {threshold:.1f})"
        else:
            explanation = f"Disable isolation (score: {isolation_score:.1f} < threshold: {threshold:.1f})"

        # Add key contributing factors
        key_factors = []
        if scenario.step_count >= self.dynamic_thresholds["step_count"].current_value:
            key_factors.append(f"High step count({scenario.step_count})")
        if (
            scenario.context_size
            >= self.dynamic_thresholds["context_size"].current_value
        ):
            key_factors.append(f"Large context({scenario.context_size})")
        if features.query_complexity_score >= 0.5:
            key_factors.append(f"Complex query({features.query_complexity_score:.1f})")
        if features.task_urgency_score >= 0.7:
            key_factors.append("Urgent task")

        if key_factors:
            explanation += f" - Key factors: {', '.join(key_factors)}"

        explanation += f" (Confidence: {confidence.confidence_score:.1f})"

        return explanation

    def _get_feature_contributions(
        self, features: AdvancedScenarioFeatures
    ) -> Dict[str, float]:
        """Get contribution of each feature to the decision"""
        scenario = features.base_scenario
        contributions = {}

        # Calculate normalized contributions
        if scenario.step_count >= self.dynamic_thresholds["step_count"].current_value:
            contributions["step_count"] = self.feature_weights["step_count"]

        if (
            scenario.context_size
            >= self.dynamic_thresholds["context_size"].current_value
        ):
            contributions["context_size"] = self.feature_weights["context_size"]

        if (
            scenario.estimated_tokens
            >= self.dynamic_thresholds["token_estimate"].current_value
        ):
            contributions["token_estimate"] = self.feature_weights["token_estimate"]

        contributions["complexity_score"] = (
            features.query_complexity_score * self.feature_weights["complexity_score"]
        )
        contributions["resource_pressure"] = (
            features.resource_pressure * self.feature_weights["resource_pressure"]
        )
        contributions["task_urgency"] = (
            features.task_urgency_score * self.feature_weights["user_urgency"]
        )

        return contributions

    def _record_decision(
        self,
        features: AdvancedScenarioFeatures,
        decision: bool,
        score: float,
        confidence: DecisionConfidence,
    ):
        """Record decision for learning and analysis"""
        decision_record = {
            "timestamp": time.time(),
            "features": features,
            "decision": decision,
            "score": score,
            "confidence": confidence,
            "threshold_values": {
                k: v.current_value for k, v in self.dynamic_thresholds.items()
            },
        }

        self.decision_history.append(decision_record)

    def _auto_tune_parameters(self):
        """Auto-tune parameters based on recent performance"""
        if len(self.decision_history) < 20:
            return

        logger.info("Starting auto-tuning of parameters")

        # Analyze recent decisions
        recent_decisions = list(self.decision_history)[-50:]

        # Calculate feature importance
        self._update_feature_importance(recent_decisions)

        # Adjust feature weights based on importance
        self._adjust_feature_weights()

        # Update dynamic thresholds based on performance feedback
        self._update_dynamic_thresholds(recent_decisions)

        logger.info(
            f"Auto-tuning completed. Updated {len(self.dynamic_thresholds)} thresholds"
        )

    def record_scenario_outcome(
        self,
        scenario: ScenarioContext,
        isolation_enabled: bool,
        execution_time: float,
        token_savings: int = 0,
        user_satisfaction: Optional[float] = None,
    ):
        """Record the outcome of a scenario for learning"""
        # Extract features for this scenario
        features = self.extract_advanced_features(scenario)

        # Create decision confidence based on execution time
        confidence = DecisionConfidence(
            confidence_score=max(0.1, min(1.0, 1.0 - execution_time / 10.0)),
            contributing_factors=["execution_time"],
            uncertainty_sources=[],
            alternative_probability=0.5,
        )

        # Record the decision
        self._record_decision(features, isolation_enabled, execution_time, confidence)

        # Update performance predictor if available
        if hasattr(self, "performance_predictor"):
            predicted_time, _ = self.performance_predictor.predict(
                features, isolation_enabled
            )
            self.performance_predictor.update_accuracy(execution_time, predicted_time)

        logger.debug(
            f"Recorded scenario outcome: isolation={isolation_enabled}, "
            f"execution_time={execution_time:.2f}, token_savings={token_savings}"
        )

    def _update_feature_importance(self, decisions: List[Dict]):
        """Update feature importance based on decision outcomes"""
        # This is a simplified importance calculation
        # In a real ML system, this would be more sophisticated

        for decision_record in decisions:
            features = decision_record["features"]
            decision_record["decision"]
            confidence = decision_record["confidence"].confidence_score

            # Weight by confidence - more confident decisions contribute more
            weight = confidence

            # Update importance for each feature
            scenario = features.base_scenario
            if scenario.step_count >= 3:
                self.feature_importance["step_count"] += weight
            if scenario.context_size >= 5000:
                self.feature_importance["context_size"] += weight
            if features.query_complexity_score >= 0.5:
                self.feature_importance["complexity_score"] += weight

    def _adjust_feature_weights(self):
        """Adjust feature weights based on importance"""
        if not self.feature_importance:
            return

        # Normalize importance scores
        max_importance = max(self.feature_importance.values())
        if max_importance == 0:
            return

        # Adjust weights (conservative adjustment)
        adjustment_rate = 0.1
        for feature, importance in self.feature_importance.items():
            if feature in self.feature_weights:
                normalized_importance = importance / max_importance
                current_weight = self.feature_weights[feature]
                target_weight = 0.5 + normalized_importance  # Range: 0.5 to 1.5

                # Gradual adjustment
                self.feature_weights[feature] = current_weight + adjustment_rate * (
                    target_weight - current_weight
                )

        logger.debug(f"Adjusted feature weights: {self.feature_weights}")

    def _update_dynamic_thresholds(self, decisions: List[Dict]):
        """Update dynamic thresholds based on recent performance"""
        # Group decisions by outcome and calculate performance feedback
        for decision_record in decisions:
            confidence = decision_record["confidence"].confidence_score

            # Use confidence as a proxy for performance
            # High confidence decisions that worked well should reinforce current thresholds
            # Low confidence decisions suggest thresholds need adjustment
            performance_feedback = confidence - 0.5  # Range: -0.5 to 0.5

            # Update each threshold
            for threshold in self.dynamic_thresholds.values():
                threshold.adjust(performance_feedback)

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "decision_count": self.decision_count,
            "feature_weights": self.feature_weights.copy(),
            "dynamic_thresholds": {
                name: {
                    "current_value": threshold.current_value,
                    "min_value": threshold.min_value,
                    "max_value": threshold.max_value,
                    "performance_history_length": len(threshold.performance_history),
                }
                for name, threshold in self.dynamic_thresholds.items()
            },
            "feature_importance": dict(self.feature_importance),
            "pattern_cache_size": len(self.pattern_cache),
            "recent_decisions": len(self.decision_history),
            "auto_tune_enabled": self.auto_tune_enabled,
            "performance_predictor_stats": self.performance_predictor.get_stats(),
        }


class PerformancePredictor:
    """Simple performance predictor for isolation decisions"""

    def __init__(self):
        self.prediction_history: List[Tuple[AdvancedScenarioFeatures, bool, float]] = []
        self.accuracy_score = 0.0

    def predict(
        self, features: AdvancedScenarioFeatures, enable_isolation: bool
    ) -> Tuple[float, float]:
        """Predict performance with and without isolation"""
        scenario = features.base_scenario

        # Simple heuristic-based prediction
        base_time = 1.0 + scenario.step_count * 0.5 + scenario.context_size / 10000

        if enable_isolation:
            # Isolation adds overhead but may save time for complex tasks
            isolation_overhead = 0.1 + features.resource_pressure * 0.2
            complexity_benefit = features.query_complexity_score * 0.3
            predicted_time = base_time + isolation_overhead - complexity_benefit
        else:
            # No isolation - potential context pollution slowdown for complex tasks
            context_pollution_penalty = (
                scenario.step_count * 0.1 + features.query_complexity_score * 0.2
            )
            predicted_time = base_time + context_pollution_penalty

        return max(0.1, predicted_time), max(0.1, base_time)

    def update_accuracy(self, actual_performance: float, predicted_performance: float):
        """Update prediction accuracy"""
        error = abs(actual_performance - predicted_performance) / max(
            actual_performance, 0.1
        )
        self.accuracy_score = self.accuracy_score * 0.9 + (1 - error) * 0.1

    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        return {
            "prediction_count": len(self.prediction_history),
            "accuracy_score": self.accuracy_score,
        }


# Global advanced enabler instance
_global_advanced_enabler: Optional[AdvancedResearcherProgressiveEnabler] = None


def get_advanced_progressive_enabler() -> AdvancedResearcherProgressiveEnabler:
    """Get global advanced progressive enabler instance"""
    global _global_advanced_enabler
    if _global_advanced_enabler is None:
        _global_advanced_enabler = AdvancedResearcherProgressiveEnabler()
    return _global_advanced_enabler


def reset_advanced_progressive_enabler():
    """Reset global advanced progressive enabler (for testing)"""
    global _global_advanced_enabler
    _global_advanced_enabler = None
