# DeerFlow Collaboration Mechanism Optimization - Conflict Resolution and Consensus Mechanism

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Conflict types"""

    DATA_INCONSISTENCY = "data_inconsistency"  # Data inconsistency
    METHOD_DISAGREEMENT = "method_disagreement"  # Method disagreement
    RESULT_CONTRADICTION = "result_contradiction"  # Result contradiction
    TOOL_INCOMPATIBILITY = "tool_incompatibility"  # Tool incompatibility
    PRIORITY_CONFLICT = "priority_conflict"  # Priority conflict


class EvidenceType(Enum):
    """Evidence types"""

    DATA_SOURCE = "data_source"  # Data source
    CALCULATION_RESULT = "calculation_result"  # Calculation result
    EXTERNAL_VALIDATION = "external_validation"  # External validation
    HISTORICAL_PRECEDENT = "historical_precedent"  # Historical precedent
    EXPERT_OPINION = "expert_opinion"  # Expert opinion


@dataclass
class Evidence:
    """Evidence"""

    evidence_id: str
    evidence_type: EvidenceType
    source: str
    content: Any
    reliability_score: float  # Reliability score 0-1
    timestamp: datetime = field(default_factory=datetime.now)
    verification_status: str = "pending"  # pending, verified, disputed


@dataclass
class ConflictingClaim:
    """Conflicting claim"""

    claim_id: str
    agent_id: str
    claim_content: str
    supporting_evidence: List[Evidence] = field(default_factory=list)
    confidence_level: float = 0.0
    reasoning: str = ""


@dataclass
class Conflict:
    """Conflict"""

    conflict_id: str
    conflict_type: ConflictType
    description: str
    conflicting_claims: List[ConflictingClaim] = field(default_factory=list)
    resolution_status: str = "unresolved"  # unresolved, resolved, escalated
    resolution_method: Optional[str] = None
    final_decision: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class EvidenceValidator:
    """Evidence validator"""

    def __init__(self):
        self.external_apis = {
            "financial_data": self._validate_financial_data,
            "scientific_data": self._validate_scientific_data,
            "statistical_calculation": self._validate_statistical_calculation,
        }

    async def validate_evidence(self, evidence: Evidence) -> Dict[str, Any]:
        """Validate evidence"""
        validation_result = {
            "is_valid": True,
            "confidence": evidence.reliability_score,
            "issues": [],
            "verification_details": {},
        }

        try:
            # Select validation method based on evidence type
            if evidence.evidence_type == EvidenceType.DATA_SOURCE:
                validation_result = await self._validate_data_source(evidence)
            elif evidence.evidence_type == EvidenceType.CALCULATION_RESULT:
                validation_result = await self._validate_calculation(evidence)
            elif evidence.evidence_type == EvidenceType.EXTERNAL_VALIDATION:
                validation_result = await self._validate_external_source(evidence)

        except Exception as e:
            logger.error(f"Evidence validation failed: {e}")
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Validation exception: {str(e)}")

        return validation_result

    async def _validate_data_source(self, evidence: Evidence) -> Dict[str, Any]:
        """Validate data source"""
        content = evidence.content
        issues = []

        # Check data integrity
        if isinstance(content, dict):
            if "data" not in content:
                issues.append("Missing data field")
            if "source_url" not in content:
                issues.append("Missing data source URL")

        # Check data freshness
        if "timestamp" in content:
            try:
                data_time = datetime.fromisoformat(content["timestamp"])
                age_hours = (datetime.now() - data_time).total_seconds() / 3600
                if age_hours > 24:
                    issues.append(f"Data expired: {age_hours:.1f} hours ago")
            except ValueError:
                issues.append("Invalid timestamp format")

        return {
            "is_valid": len(issues) == 0,
            "confidence": max(0.1, evidence.reliability_score - len(issues) * 0.2),
            "issues": issues,
            "verification_details": {"data_checks": len(issues) == 0},
        }

    async def _validate_calculation(self, evidence: Evidence) -> Dict[str, Any]:
        """Validate calculation result"""
        content = evidence.content
        issues = []

        # Re-execute calculation validation
        if isinstance(content, dict) and "calculation" in content:
            try:
                # Simplified calculation validation logic
                calc_data = content["calculation"]
                if (
                    "formula" in calc_data
                    and "inputs" in calc_data
                    and "result" in calc_data
                ):
                    # Actual calculation verification can be implemented here
                    # result = eval(calc_data["formula"], calc_data["inputs"])
                    # if abs(result - calc_data["result"]) > 0.01:
                    #     issues.append("Calculation result mismatch")
                    pass
                else:
                    issues.append("Incomplete calculation information")
            except Exception as e:
                issues.append(f"Calculation verification failed: {str(e)}")

        return {
            "is_valid": len(issues) == 0,
            "confidence": max(0.1, evidence.reliability_score - len(issues) * 0.3),
            "issues": issues,
            "verification_details": {"calculation_verified": len(issues) == 0},
        }

    async def _validate_external_source(self, evidence: Evidence) -> Dict[str, Any]:
        """Validate external source"""
        content = evidence.content
        issues = []

        # Check external source credibility
        if isinstance(content, dict) and "source" in content:
            source = content["source"]
            trusted_sources = [
                "bloomberg",
                "reuters",
                "wind",
                "edgar",
                "pubmed",
                "arxiv",
            ]

            is_trusted = any(trusted in source.lower() for trusted in trusted_sources)
            if not is_trusted:
                issues.append("Untrusted data source")

        return {
            "is_valid": len(issues) == 0,
            "confidence": evidence.reliability_score if len(issues) == 0 else 0.3,
            "issues": issues,
            "verification_details": {"source_trusted": len(issues) == 0},
        }

    async def _validate_financial_data(self, data: Dict[str, Any]) -> bool:
        """Validate financial data (simulate calling Wind terminal or EDGAR)"""
        # Simulate external API call
        await asyncio.sleep(0.1)
        return True

    async def _validate_scientific_data(self, data: Dict[str, Any]) -> bool:
        """Validate scientific data (simulate calling PubMed or ArXiv)"""
        await asyncio.sleep(0.1)
        return True

    async def _validate_statistical_calculation(
        self, calculation: Dict[str, Any]
    ) -> bool:
        """Validate statistical calculation"""
        await asyncio.sleep(0.1)
        return True


class ConflictDetector:
    """Conflict detector"""

    def __init__(self):
        self.similarity_threshold = 0.8
        self.contradiction_patterns = [
            {
                "pattern": ["increase", "decrease"],
                "type": ConflictType.RESULT_CONTRADICTION,
            },
            {
                "pattern": ["positive", "negative"],
                "type": ConflictType.RESULT_CONTRADICTION,
            },
            {
                "pattern": ["profitable", "loss"],
                "type": ConflictType.DATA_INCONSISTENCY,
            },
        ]

    def detect_conflicts(self, claims: List[ConflictingClaim]) -> List[Conflict]:
        """Detect conflicts"""
        conflicts = []

        # Compare all claims pairwise
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                conflict = self._compare_claims(claims[i], claims[j])
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def _compare_claims(
        self, claim1: ConflictingClaim, claim2: ConflictingClaim
    ) -> Optional[Conflict]:
        """Compare two claims"""
        # Check direct contradiction
        contradiction_type = self._check_direct_contradiction(
            claim1.claim_content, claim2.claim_content
        )

        if contradiction_type:
            conflict_id = self._generate_conflict_id(claim1, claim2)
            return Conflict(
                conflict_id=conflict_id,
                conflict_type=contradiction_type,
                description=f"Agent {claim1.agent_id} and {claim2.agent_id} have different conclusions on the same issue",
                conflicting_claims=[claim1, claim2],
            )

        # Check data inconsistency
        if self._check_data_inconsistency(claim1, claim2):
            conflict_id = self._generate_conflict_id(claim1, claim2)
            return Conflict(
                conflict_id=conflict_id,
                conflict_type=ConflictType.DATA_INCONSISTENCY,
                description=f"Agent {claim1.agent_id} and {claim2.agent_id} used inconsistent data",
                conflicting_claims=[claim1, claim2],
            )

        return None

    def _check_direct_contradiction(
        self, content1: str, content2: str
    ) -> Optional[ConflictType]:
        """Check direct contradiction"""
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        for pattern in self.contradiction_patterns:
            words = pattern["pattern"]
            if any(word in content1_lower for word in words) and any(
                word in content2_lower for word in words
            ):
                # Check if it's a real contradiction (simplified logic)
                if words[0] in content1_lower and words[1] in content2_lower:
                    return pattern["type"]
                elif words[1] in content1_lower and words[0] in content2_lower:
                    return pattern["type"]

        return None

    def _check_data_inconsistency(
        self, claim1: ConflictingClaim, claim2: ConflictingClaim
    ) -> bool:
        """Check data inconsistency"""
        # Compare data sources of supporting evidence
        sources1 = set()
        sources2 = set()

        for evidence in claim1.supporting_evidence:
            if evidence.evidence_type == EvidenceType.DATA_SOURCE:
                sources1.add(evidence.source)

        for evidence in claim2.supporting_evidence:
            if evidence.evidence_type == EvidenceType.DATA_SOURCE:
                sources2.add(evidence.source)

        # If different data sources are used to reach different conclusions, there may be data inconsistency
        return (
            len(sources1.intersection(sources2)) == 0
            and len(sources1) > 0
            and len(sources2) > 0
        )

    def _generate_conflict_id(
        self, claim1: ConflictingClaim, claim2: ConflictingClaim
    ) -> str:
        """Generate conflict ID"""
        combined = f"{claim1.claim_id}_{claim2.claim_id}_{datetime.now().isoformat()}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]


class ConsensusBuilder:
    """Consensus builder"""

    def __init__(self):
        self.evidence_validator = EvidenceValidator()
        self.voting_weights = {
            "evidence_quality": 0.4,
            "agent_reputation": 0.3,
            "verification_status": 0.3,
        }

    async def resolve_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict"""
        logger.info(f"Starting to resolve conflict: {conflict.conflict_id}")

        resolution_methods = [
            self._evidence_based_resolution,
            self._voting_based_resolution,
            self._external_verification_resolution,
        ]

        for method in resolution_methods:
            try:
                result = await method(conflict)
                if result["success"]:
                    conflict.resolution_status = "resolved"
                    conflict.resolution_method = result["method"]
                    conflict.final_decision = result["decision"]
                    return result
            except Exception as e:
                logger.error(f"resolution methods Error in {method.__name__}: {e}")
                continue

        # If all methods fail, mark as requiring human intervention
        conflict.resolution_status = "escalated"
        return {
            "success": False,
            "method": "human_escalation",
            "decision": "Manual intervention required to resolve conflict",
            "reason": "All automatic resolution methods failed",
        }

    async def _evidence_based_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """Evidence-based resolution"""
        logger.info(
            f"Using evidence analysis to resolve conflict: {conflict.conflict_id}"
        )

        claim_scores = []

        for claim in conflict.conflicting_claims:
            evidence_score = 0.0
            total_evidence = len(claim.supporting_evidence)

            if total_evidence == 0:
                evidence_score = 0.1  # Claims without evidence get very low scores
            else:
                for evidence in claim.supporting_evidence:
                    validation = await self.evidence_validator.validate_evidence(
                        evidence
                    )
                    evidence_score += validation["confidence"]
                evidence_score /= total_evidence

            claim_scores.append(
                {
                    "claim_id": claim.claim_id,
                    "agent_id": claim.agent_id,
                    "evidence_score": evidence_score,
                    "confidence": claim.confidence_level,
                    "total_score": evidence_score * 0.7 + claim.confidence_level * 0.3,
                }
            )

        # Select the claim with the highest score
        best_claim = max(claim_scores, key=lambda x: x["total_score"])

        if best_claim["total_score"] > 0.6:
            return {
                "success": True,
                "method": "evidence_based",
                "decision": f"Support agent {best_claim['agent_id']}'s claim",
                "winning_claim": best_claim,
                "all_scores": claim_scores,
                "reason": f"Highest evidence quality, total score: {best_claim['total_score']:.2f}",
            }
        else:
            return {
                "success": False,
                "reason": "Evidence quality of all claims is insufficient to make a decision",
            }

    async def _voting_based_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """Voting-based resolution"""
        logger.info(
            f"Using voting mechanism to resolve conflict: {conflict.conflict_id}"
        )

        # Simulate voting on multiple evaluation dimensions
        voting_dimensions = [
            "Data Quality",
            "Method Reasonableness",
            "Result Credibility",
            "Logical Consistency",
        ]
        votes = {}

        for claim in conflict.conflicting_claims:
            votes[claim.agent_id] = 0

            # Vote for each dimension (simplified voting logic)
            for dimension in voting_dimensions:
                score = self._evaluate_claim_dimension(claim, dimension)
                votes[claim.agent_id] += score

        # Determine the winner
        winner = max(votes.items(), key=lambda x: x[1])

        if winner[1] > sum(votes.values()) * 0.4:  # Need more than 40% of votes
            return {
                "success": True,
                "method": "voting_based",
                "decision": f"Voting results support agent {winner[0]}",
                "vote_details": votes,
                "reason": f"Received {winner[1]:.1f} votes, accounting for {winner[1]/sum(votes.values())*100:.1f}% of total votes",
            }
        else:
            return {
                "success": False,
                "reason": "Voting results are too close to determine a clear winner",
            }

    async def _external_verification_resolution(
        self, conflict: Conflict
    ) -> Dict[str, Any]:
        """External verification resolution"""
        logger.info(
            f"Using external verification to resolve conflict: {conflict.conflict_id}"
        )

        verification_results = []

        for claim in conflict.conflicting_claims:
            # Select verifiable evidence for external verification
            verifiable_evidence = [
                e
                for e in claim.supporting_evidence
                if e.evidence_type
                in [EvidenceType.DATA_SOURCE, EvidenceType.CALCULATION_RESULT]
            ]

            if not verifiable_evidence:
                verification_results.append(
                    {
                        "agent_id": claim.agent_id,
                        "verified": False,
                        "reason": "No verifiable evidence",
                    }
                )
                continue

            # Execute external verification
            verification_success = True
            verification_details = []

            for evidence in verifiable_evidence:
                try:
                    # Simulate external API verification
                    if evidence.source in ["wind", "bloomberg", "edgar"]:
                        verified = (
                            await self.evidence_validator._validate_financial_data(
                                evidence.content
                            )
                        )
                    elif evidence.source in ["pubmed", "arxiv"]:
                        verified = (
                            await self.evidence_validator._validate_scientific_data(
                                evidence.content
                            )
                        )
                    else:
                        verified = True  # Assume other sources are verified

                    verification_details.append(
                        {"evidence_id": evidence.evidence_id, "verified": verified}
                    )

                    if not verified:
                        verification_success = False

                except Exception as e:
                    verification_success = False
                    verification_details.append(
                        {
                            "evidence_id": evidence.evidence_id,
                            "verified": False,
                            "error": str(e),
                        }
                    )

            verification_results.append(
                {
                    "agent_id": claim.agent_id,
                    "verified": verification_success,
                    "details": verification_details,
                }
            )

        # Select successfully verified claims
        verified_claims = [r for r in verification_results if r["verified"]]

        if len(verified_claims) == 1:
            winner = verified_claims[0]
            return {
                "success": True,
                "method": "external_verification",
                "decision": f"External verification supports agent {winner['agent_id']}",
                "verification_results": verification_results,
                "reason": "Only one claim passed external verification",
            }
        elif len(verified_claims) > 1:
            return {
                "success": False,
                "reason": "Multiple claims passed external verification, further analysis needed",
            }
        else:
            return {
                "success": False,
                "reason": "No claims passed external verification",
            }

    def _evaluate_claim_dimension(
        self, claim: ConflictingClaim, dimension: str
    ) -> float:
        """Evaluate claim score in specific dimension"""
        # Simplified evaluation logic
        base_score = claim.confidence_level

        if dimension == "Data Quality":
            data_evidence_count = sum(
                1
                for e in claim.supporting_evidence
                if e.evidence_type == EvidenceType.DATA_SOURCE
            )
            return base_score * (1 + data_evidence_count * 0.1)
        elif dimension == "Method Reasonableness":
            return (
                base_score * 0.8 + len(claim.reasoning) / 1000
            )  # Reasoning length as reasonableness indicator
        elif dimension == "Result Credibility":
            return (
                base_score
                * sum(e.reliability_score for e in claim.supporting_evidence)
                / max(1, len(claim.supporting_evidence))
            )
        elif dimension == "Logical Consistency":
            # Simplified logical consistency check
            return (
                base_score
                if "because" in claim.reasoning or "therefore" in claim.reasoning
                else base_score * 0.8
            )

        return base_score


class ConflictResolutionSystem:
    """Conflict resolution system"""

    def __init__(self):
        self.conflict_detector = ConflictDetector()
        self.consensus_builder = ConsensusBuilder()
        self.active_conflicts: Dict[str, Conflict] = {}
        self.resolution_history: List[Dict[str, Any]] = []

    async def process_claims(self, claims: List[ConflictingClaim]) -> Dict[str, Any]:
        """Process claims and resolve conflicts"""
        logger.info(f"Processing {len(claims)} claims")

        # Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(claims)
        logger.info(f"Detected {len(conflicts)} conflicts")

        resolution_results = []

        for conflict in conflicts:
            self.active_conflicts[conflict.conflict_id] = conflict

            # Resolve conflict
            resolution = await self.consensus_builder.resolve_conflict(conflict)
            resolution["conflict_id"] = conflict.conflict_id
            resolution_results.append(resolution)

            # Record resolution history
            self.resolution_history.append(
                {
                    "conflict_id": conflict.conflict_id,
                    "conflict_type": conflict.conflict_type.value,
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "total_claims": len(claims),
            "conflicts_detected": len(conflicts),
            "resolutions": resolution_results,
            "unresolved_conflicts": len(
                [c for c in conflicts if c.resolution_status == "unresolved"]
            ),
            "escalated_conflicts": len(
                [c for c in conflicts if c.resolution_status == "escalated"]
            ),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "active_conflicts": len(self.active_conflicts),
            "total_resolutions": len(self.resolution_history),
            "success_rate": len(
                [r for r in self.resolution_history if r["resolution"]["success"]]
            )
            / max(1, len(self.resolution_history)),
            "recent_conflicts": (
                self.resolution_history[-5:] if self.resolution_history else []
            ),
        }


# Usage example
async def demo_conflict_resolution():
    """Demonstrate conflict resolution system"""
    system = ConflictResolutionSystem()

    # Create conflicting claims
    claims = [
        ConflictingClaim(
            claim_id="claim_001",
            agent_id="financial_analyst",
            claim_content="Company revenue increased 15% year-over-year, excellent performance",
            supporting_evidence=[
                Evidence(
                    evidence_id="ev_001",
                    evidence_type=EvidenceType.DATA_SOURCE,
                    source="wind",
                    content={"revenue_growth": 0.15, "source_url": "wind://data"},
                    reliability_score=0.9,
                )
            ],
            confidence_level=0.8,
            reasoning="Based on Wind terminal financial data analysis",
        ),
        ConflictingClaim(
            claim_id="claim_002",
            agent_id="research_specialist",
            claim_content="Company revenue shows declining trend, situation is not optimistic",
            supporting_evidence=[
                Evidence(
                    evidence_id="ev_002",
                    evidence_type=EvidenceType.CALCULATION_RESULT,
                    source="internal_calculation",
                    content={
                        "calculation": {
                            "formula": "growth_rate",
                            "inputs": {},
                            "result": -0.05,
                        }
                    },
                    reliability_score=0.7,
                )
            ],
            confidence_level=0.7,
            reasoning="Conclusion based on internal calculations",
        ),
    ]

    # Process conflicts
    results = await system.process_claims(claims)

    print("Conflict resolution results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Display system status
    status = system.get_system_status()
    print("\nSystem status:")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(demo_conflict_resolution())
