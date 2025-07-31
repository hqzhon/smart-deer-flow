# DeerFlow Report Quality Optimization - Critical Thinking Injection Mechanism

import logging
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from .i18n import get_i18n_manager, Language

logger = logging.getLogger(__name__)


class LimitationType(Enum):
    """Data Limitation Types"""

    SAMPLE_SIZE = "sample_size"
    TIME_PERIOD = "time_period"
    GEOGRAPHIC_SCOPE = "geographic_scope"
    METHODOLOGY = "methodology"
    DATA_QUALITY = "data_quality"
    BIAS = "bias"
    EXTERNAL_VALIDITY = "external_validity"
    STATISTICAL_POWER = "statistical_power"


class ConfidenceLevel(Enum):
    """Statistical Confidence Levels"""

    VERY_HIGH = "very_high"  # 95%+
    HIGH = "high"  # 80-95%
    MEDIUM = "medium"  # 60-80%
    LOW = "low"  # 40-60%
    VERY_LOW = "very_low"  # <40%


class EvidenceQuality(Enum):
    """Evidence Quality Assessment"""

    STRONG = "strong"  # Multiple verification, authoritative sources
    MODERATE = "moderate"  # Partial verification, credible sources
    WEAK = "weak"  # Single source, unverified
    INSUFFICIENT = "insufficient"  # Insufficient evidence


@dataclass
class DataLimitation:
    """Data Limitation Assessment"""

    limitation_id: str
    limitation_type: LimitationType
    description: str
    impact_level: str  # high, medium, low
    mitigation_suggestions: List[str] = field(default_factory=list)
    affects_conclusions: bool = True


@dataclass
class ConfidenceAnnotation:
    """Statistical Confidence Annotation"""

    statement: str
    confidence_level: ConfidenceLevel
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    limitations: List[DataLimitation] = field(default_factory=list)
    alternative_interpretations: List[str] = field(default_factory=list)


@dataclass
class BiasAssessment:
    """Cognitive Bias Assessment"""

    bias_type: str
    severity: str  # high, medium, low
    description: str
    potential_impact: str
    mitigation_measures: List[str] = field(default_factory=list)


@dataclass
class CriticalAnalysis:
    """Critical Thinking Analysis Results"""

    analysis_id: str
    original_content: str
    confidence_annotations: List[ConfidenceAnnotation] = field(default_factory=list)
    identified_limitations: List[DataLimitation] = field(default_factory=list)
    bias_assessments: List[BiasAssessment] = field(default_factory=list)
    enhanced_content: str = ""
    quality_score: float = 0.0


class LimitationDetector:
    """Data Limitation Detection Engine"""

    def __init__(self, language: Language = Language.ZH_CN):
        self.language = language
        self.i18n = get_i18n_manager()
        self.i18n.set_language(language)

        # Detection patterns for different languages
        self.detection_patterns = {
            Language.ZH_CN: {
                LimitationType.SAMPLE_SIZE: [
                    r"(\d+)\s*(个|名|家|只)\s*(样本|案例|公司|股票)",  # N samples/cases/companies/stocks
                    r"基于\s*(\d+)\s*(个|项|例)",  # Based on N items/cases
                    r"(\d+)\s*(subjects?|participants?|companies?)",
                ],
                LimitationType.TIME_PERIOD: [
                    r"(\d{4})\s*年\s*数据",  # YYYY year data
                    r"近\s*(\d+)\s*(年|月|日)",  # Recent N years/months/days
                    r"(\d{4})\s*-\s*(\d{4})",
                    r"(Q[1-4])\s*季度",  # Q1-Q4 quarter
                ],
                LimitationType.GEOGRAPHIC_SCOPE: [
                    r"(中国|美国|欧洲|亚洲)\s*(市场|地区)",  # China/US/Europe/Asia market/region
                    r"仅限.*?地区",  # Limited to ... region
                    r"局限于.*?范围",  # Confined to ... scope
                ],
                LimitationType.METHODOLOGY: [
                    r"使用.*?方法",  # Using ... method
                    r"基于.*?模型",  # Based on ... model
                    r"通过.*?分析",  # Through ... analysis
                ],
            },
            Language.EN_US: {
                LimitationType.SAMPLE_SIZE: [
                    r"(\d+)\s*(subjects?|participants?|companies?|samples?)",
                    r"based\s+on\s+(\d+)\s*(cases?|examples?)",
                    r"sample\s+size\s+of\s+(\d+)",
                ],
                LimitationType.TIME_PERIOD: [
                    r"(\d{4})\s*data",
                    r"recent\s+(\d+)\s*(years?|months?|days?)",
                    r"(\d{4})\s*-\s*(\d{4})",
                    r"(Q[1-4])\s*quarter",
                ],
                LimitationType.GEOGRAPHIC_SCOPE: [
                    r"(China|US|Europe|Asia)\s*(market|region)",
                    r"limited\s+to.*?region",
                    r"confined\s+to.*?area",
                ],
                LimitationType.METHODOLOGY: [
                    r"using.*?method",
                    r"based\s+on.*?model",
                    r"through.*?analysis",
                ],
            },
        }

        self.quality_indicators = {
            "small_sample": {"threshold": 30, "impact": "high"},
            "short_period": {"threshold": 1, "impact": "medium"},  # 1 year threshold
            "single_source": {"impact": "medium"},
            "no_control_group": {"impact": "high"},
        }

    def set_language(self, language: Language):
        """Set language for the limitation detector"""
        self.language = language
        self.i18n.set_language(language)

    def detect_limitations(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[DataLimitation]:
        """Detect data limitations in content and metadata"""
        limitations = []

        # Detect sample size limitations
        sample_limitations = self._detect_sample_limitations(content)
        limitations.extend(sample_limitations)

        # Detect time range limitations
        time_limitations = self._detect_time_limitations(content)
        limitations.extend(time_limitations)

        # Detect geographic scope limitations
        geo_limitations = self._detect_geographic_limitations(content)
        limitations.extend(geo_limitations)

        # Detect methodology limitations
        method_limitations = self._detect_methodology_limitations(content)
        limitations.extend(method_limitations)

        # Detect additional limitations based on metadata
        meta_limitations = self._detect_metadata_limitations(metadata)
        limitations.extend(meta_limitations)

        return limitations

    def _detect_sample_limitations(self, content: str) -> List[DataLimitation]:
        """Detect sample size limitations in the content"""
        limitations = []

        patterns = self.detection_patterns.get(self.language, {}).get(
            LimitationType.SAMPLE_SIZE, []
        )
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        sample_size = int(match[0])
                    else:
                        sample_size = int(match)

                    if (
                        sample_size
                        < self.quality_indicators["small_sample"]["threshold"]
                    ):
                        limitations.append(
                            DataLimitation(
                                limitation_id=str(uuid.uuid4())[:8],
                                limitation_type=LimitationType.SAMPLE_SIZE,
                                description=self.i18n.get_text(
                                    "limitation_descriptions",
                                    "small_sample_size",
                                    sample_size=sample_size,
                                ),
                                impact_level="high" if sample_size < 10 else "medium",
                                mitigation_suggestions=[
                                    self.i18n.get_text(
                                        "mitigation_suggestions", "increase_sample_size"
                                    ),
                                    self.i18n.get_text(
                                        "mitigation_suggestions", "meta_analysis"
                                    ),
                                    self.i18n.get_text(
                                        "mitigation_suggestions", "specify_scope"
                                    ),
                                ],
                            )
                        )
                except ValueError:
                    continue

        return limitations

    def _detect_time_limitations(self, content: str) -> List[DataLimitation]:
        """Detect time range and temporal limitations"""
        limitations = []

        # Detect short time span - language-specific patterns
        if self.language == Language.ZH_CN:
            short_period_patterns = [
                r"(\d+)\s*个?月",  # Chinese: N months
                r"(\d+)\s*季度",  # Chinese: N quarters
                r"近期",  # Chinese: recent period
                r"最近",  # Chinese: recently
            ]
            year_pattern = r"(\d{4})\s*年"  # Chinese: YYYY year
        else:  # English
            short_period_patterns = [
                r"(\d+)\s*months?",
                r"(\d+)\s*quarters?",
                r"recent\s*period",
                r"lately",
            ]
            year_pattern = r"(\d{4})\s*(data|year)"

        for pattern in short_period_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                limitations.append(
                    DataLimitation(
                        limitation_id=str(uuid.uuid4())[:8],
                        limitation_type=LimitationType.TIME_PERIOD,
                        description=self.i18n.get_text(
                            "limitation_descriptions", "short_time_span"
                        ),
                        impact_level="medium",
                        mitigation_suggestions=[
                            self.i18n.get_text(
                                "mitigation_suggestions", "extend_time_range"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions", "explain_short_term_limits"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions", "compare_historical_trends"
                            ),
                        ],
                    )
                )
                break

        # Detect data timeliness issues
        years = re.findall(year_pattern, content)
        if years:
            if self.language == Language.ZH_CN:
                latest_year = max(int(year) for year in years)
            else:
                latest_year = max(
                    int(match[0]) if isinstance(match, tuple) else int(match)
                    for match in years
                )
            current_year = datetime.now().year

            if current_year - latest_year > 2:
                limitations.append(
                    DataLimitation(
                        limitation_id=str(uuid.uuid4())[:8],
                        limitation_type=LimitationType.TIME_PERIOD,
                        description=self.i18n.get_text(
                            "limitation_descriptions", "outdated_data", year=latest_year
                        ),
                        impact_level="medium",
                        mitigation_suggestions=[
                            self.i18n.get_text("mitigation_suggestions", "update_data"),
                            self.i18n.get_text(
                                "mitigation_suggestions", "explain_timeliness_limits"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions", "analyze_time_impact"
                            ),
                        ],
                    )
                )

        return limitations

    def _detect_geographic_limitations(self, content: str) -> List[DataLimitation]:
        """Detect geographic scope and regional limitations"""
        limitations = []

        # Language-specific geographic indicators
        if self.language == Language.ZH_CN:
            geographic_indicators = [
                "仅限中国",  # Chinese: limited to China only
                "仅在美国",  # Chinese: only in US
                "局限于",  # Chinese: confined to
                "限于.*地区",  # Chinese: limited to ... region
                "单一市场",  # Chinese: single market
                "特定地区",  # Chinese: specific region
                "区域性",  # Chinese: regional
            ]
        else:  # English
            geographic_indicators = [
                "limited to China",
                "only in US",
                "confined to",
                "restricted to.*region",
                "single market",
                "specific region",
                "regional",
            ]

        for indicator in geographic_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                limitations.append(
                    DataLimitation(
                        limitation_id=str(uuid.uuid4())[:8],
                        limitation_type=LimitationType.GEOGRAPHIC_SCOPE,
                        description=self.i18n.get_text(
                            "limitation_descriptions", "geographic_limitation"
                        ),
                        impact_level="medium",
                        mitigation_suggestions=[
                            self.i18n.get_text(
                                "mitigation_suggestions", "extend_geographic_scope"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions", "cross_regional_study"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions",
                                "clarify_geographic_applicability",
                            ),
                        ],
                    )
                )
                break

        return limitations

    def _detect_methodology_limitations(self, content: str) -> List[DataLimitation]:
        """Detect research methodology limitations"""
        limitations = []

        # Language-specific methodology indicators
        if self.language == Language.ZH_CN:
            single_method_indicators = [
                "仅使用",  # Chinese: only use
                "只采用",  # Chinese: solely adopt
                "单一方法",  # Chinese: single method
                "基于.*模型",  # Chinese: based on ... model
            ]
        else:  # English
            single_method_indicators = [
                "only use",
                "solely adopt",
                "single method",
                "based on.*model",
            ]

        for indicator in single_method_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                limitations.append(
                    DataLimitation(
                        limitation_id=str(uuid.uuid4())[:8],
                        limitation_type=LimitationType.METHODOLOGY,
                        description=self.i18n.get_text(
                            "limitation_descriptions", "single_methodology"
                        ),
                        impact_level="medium",
                        mitigation_suggestions=[
                            self.i18n.get_text(
                                "mitigation_suggestions", "multiple_methods"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions", "explain_methodology_limits"
                            ),
                            self.i18n.get_text(
                                "mitigation_suggestions", "sensitivity_analysis"
                            ),
                        ],
                    )
                )
                break

        return limitations

    def _detect_metadata_limitations(
        self, metadata: Dict[str, Any]
    ) -> List[DataLimitation]:
        """Detect limitations based on research metadata"""
        limitations = []

        # Detect single data source dependency
        data_sources = metadata.get("data_sources", [])
        if len(data_sources) == 1:
            limitations.append(
                DataLimitation(
                    limitation_id=str(uuid.uuid4())[:8],
                    limitation_type=LimitationType.DATA_QUALITY,
                    description=self.i18n.get_text(
                        "limitation_descriptions", "single_data_source"
                    ),
                    impact_level="medium",
                    mitigation_suggestions=[
                        self.i18n.get_text(
                            "mitigation_suggestions", "multiple_data_sources"
                        ),
                        self.i18n.get_text(
                            "mitigation_suggestions", "assess_source_reliability"
                        ),
                        self.i18n.get_text(
                            "mitigation_suggestions", "explain_single_source_limits"
                        ),
                    ],
                )
            )

        # Detect experimental design flaws
        if not metadata.get("has_control_group", True):
            limitations.append(
                DataLimitation(
                    limitation_id=str(uuid.uuid4())[:8],
                    limitation_type=LimitationType.METHODOLOGY,
                    description=self.i18n.get_text(
                        "limitation_descriptions", "no_control_group"
                    ),
                    impact_level="high",
                    mitigation_suggestions=[
                        self.i18n.get_text(
                            "mitigation_suggestions", "setup_control_groups"
                        ),
                        self.i18n.get_text(
                            "mitigation_suggestions", "quasi_experimental_design"
                        ),
                        self.i18n.get_text(
                            "mitigation_suggestions", "clarify_correlation_causation"
                        ),
                    ],
                )
            )

        return limitations


class ConfidenceAssessor:
    """Statistical Confidence Assessment Engine"""

    def __init__(self, language: Language = Language.ZH_CN):
        self.language = language
        self.i18n = get_i18n_manager()
        self.i18n.set_language(language)

        # Language-specific confidence level indicators
        self.confidence_indicators = {
            Language.ZH_CN: {
                ConfidenceLevel.VERY_HIGH: [
                    "多项研究证实",  # Chinese: multiple studies confirm
                    "大规模随机对照试验",  # Chinese: large-scale randomized controlled trial
                    "元分析显示",  # Chinese: meta-analysis shows
                    "权威机构确认",  # Chinese: authoritative institution confirms
                    "结论一致",  # Chinese: consistent conclusions
                ],
                ConfidenceLevel.HIGH: [
                    "多个数据源",  # Chinese: multiple data sources
                    "反复验证",  # Chinese: repeated verification
                    "统计显著",  # Chinese: statistically significant
                    "权威来源",  # Chinese: authoritative sources
                    "长期跟踪",  # Chinese: long-term tracking
                ],
                ConfidenceLevel.MEDIUM: [
                    "初步研究",  # Chinese: preliminary research
                    "部分证据",  # Chinese: partial evidence
                    "样本有限",  # Chinese: limited sample
                    "短期观察",  # Chinese: short-term observation
                    "单一研究",  # Chinese: single study
                ],
                ConfidenceLevel.LOW: [
                    "theoretical speculation",
                    "informal observation",
                    "anecdotal evidence",
                    "expert opinion",
                    "preliminary analysis",
                ],
                ConfidenceLevel.VERY_LOW: [
                    "speculation",
                    "hypothesis",
                    "possibly",
                    "perhaps",
                    "no conclusion yet",
                    "insufficient evidence",
                ],
            },
            Language.EN_US: {
                ConfidenceLevel.VERY_HIGH: [
                    "multiple studies confirm",
                    "large-scale randomized controlled trial",
                    "meta-analysis shows",
                    "authoritative institution confirms",
                    "consistent conclusions",
                ],
                ConfidenceLevel.HIGH: [
                    "multiple data sources",
                    "repeated verification",
                    "statistically significant",
                    "authoritative sources",
                    "long-term tracking",
                ],
                ConfidenceLevel.MEDIUM: [
                    "preliminary research",
                    "partial evidence",
                    "limited sample",
                    "short-term observation",
                    "single study",
                ],
                ConfidenceLevel.LOW: [
                    "theoretical speculation",
                    "informal observation",
                    "anecdotal evidence",
                    "expert opinion",
                    "preliminary analysis",
                ],
                ConfidenceLevel.VERY_LOW: [
                    "speculation",
                    "hypothesis",
                    "possibly",
                    "perhaps",
                    "no conclusion yet",
                    "insufficient evidence",
                ],
            },
        }

    def set_language(self, language: Language):
        """Set language for the confidence assessor"""
        self.language = language
        self.i18n.set_language(language)

    def assess_confidence(
        self, statement: str, supporting_evidence: List[str]
    ) -> ConfidenceAnnotation:
        """Assess confidence level of a statement based on evidence"""
        confidence_level = self._determine_confidence_level(
            statement, supporting_evidence
        )
        reasoning = self._generate_confidence_reasoning(
            statement, supporting_evidence, confidence_level
        )

        return ConfidenceAnnotation(
            statement=statement,
            confidence_level=confidence_level,
            reasoning=reasoning,
            supporting_evidence=supporting_evidence,
        )

    def _determine_confidence_level(
        self, statement: str, evidence: List[str]
    ) -> ConfidenceLevel:
        """Determine confidence level based on statement and evidence"""
        statement_lower = statement.lower()
        evidence_text = " ".join(evidence).lower()

        # Check indicators for different confidence levels using current language
        current_indicators = self.confidence_indicators[self.language]
        for level, indicators in current_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower or indicator in evidence_text:
                    return level

        # Fallback: determine based on evidence quantity and quality
        evidence_count = len(evidence)
        if evidence_count >= 3:
            return ConfidenceLevel.HIGH
        elif evidence_count >= 2:
            return ConfidenceLevel.MEDIUM
        elif evidence_count >= 1:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_confidence_reasoning(
        self, statement: str, evidence: List[str], level: ConfidenceLevel
    ) -> str:
        """Generate reasoning explanation for confidence level"""
        evidence_count = len(evidence)

        reasoning_templates = {
            ConfidenceLevel.VERY_HIGH: self.i18n.get_text(
                "confidence_reasoning", "very_high"
            ),
            ConfidenceLevel.HIGH: self.i18n.get_text("confidence_reasoning", "high"),
            ConfidenceLevel.MEDIUM: self.i18n.get_text(
                "confidence_reasoning", "medium"
            ),
            ConfidenceLevel.LOW: self.i18n.get_text("confidence_reasoning", "low"),
            ConfidenceLevel.VERY_LOW: self.i18n.get_text(
                "confidence_reasoning", "very_low"
            ),
        }

        template = reasoning_templates.get(
            level, self.i18n.get_text("confidence_reasoning", "error")
        )
        return template.format(count=evidence_count)


class BiasDetector:
    """Cognitive Bias Detection Engine"""

    def __init__(self, language: Language = Language.ZH_CN):
        self.language = language
        self.i18n = get_i18n_manager()
        self.i18n.set_language(language)

        # Language-specific bias detection patterns
        self.bias_patterns = {
            Language.ZH_CN: {
                "confirmation_bias": [
                    "只考虑.*支持",  # Chinese: only consider supporting
                    "忽略.*相反",  # Chinese: ignore contrary
                    "选择性.*使用",  # Chinese: selective use
                    "只关注.*积极",  # Chinese: only focus on positive
                    "排除.*消极",  # Chinese: exclude negative
                    "有利.*证据",  # Chinese: favorable evidence
                ],
                "survivorship_bias": [
                    "成功案例",  # Chinese: success cases
                    "幸存者",  # Chinese: survivors
                    "只看.*存活",  # Chinese: only look at surviving
                    "忽略.*失败",  # Chinese: ignore failures
                    "排除.*退市",  # Chinese: exclude delisted
                    "成功.*企业",  # Chinese: successful companies
                ],
                "availability_bias": [
                    "最近案例",  # Chinese: recent cases
                    "容易想到",  # Chinese: easy to think of
                    "媒体报道",  # Chinese: media reports
                    "印象深刻",  # Chinese: impressive
                    "知名案例",  # Chinese: well-known cases
                    "热门.*事件",  # Chinese: hot events
                ],
                "anchoring_bias": [
                    "基于.*基准",  # Chinese: based on benchmark
                    "参考.*价格",  # Chinese: reference price
                    "基于.*预期",  # Chinese: based on expectation
                    "锚定.*价值",  # Chinese: anchored value
                    "受.*影响",  # Chinese: influenced by
                    "参照.*标准",  # Chinese: reference standard
                ],
            },
            Language.EN_US: {
                "confirmation_bias": [
                    "only consider.*support",
                    "ignore.*contrary",
                    "selective.*use",
                    "only focus.*positive",
                    "exclude.*negative",
                ],
                "survivorship_bias": [
                    "success cases",
                    "survivors",
                    "only look.*surviving",
                    "ignore.*failure",
                    "exclude.*delisted",
                ],
                "availability_bias": [
                    "recent cases",
                    "easy to think",
                    "media reports",
                    "impressive",
                    "well-known cases",
                ],
                "anchoring_bias": [
                    "based on.*benchmark",
                    "reference.*price",
                    "based on.*expectation",
                    "anchored.*value",
                    "influenced by.*",
                ],
            },
        }

    def set_language(self, language: Language):
        """Set language for the bias detector"""
        self.language = language
        self.i18n.set_language(language)

    def detect_biases(
        self, content: str, methodology: Dict[str, Any]
    ) -> List[BiasAssessment]:
        """Detect cognitive biases in content and methodology"""
        biases = []

        # Text pattern detection using current language patterns
        current_patterns = self.bias_patterns[self.language]
        for bias_type, patterns in current_patterns.items():
            if self._check_bias_patterns(content, patterns):
                biases.append(self._create_bias_assessment(bias_type, content))

        # Methodology bias detection
        method_biases = self._detect_methodology_biases(methodology)
        biases.extend(method_biases)

        return biases

    def _check_bias_patterns(self, content: str, patterns: List[str]) -> bool:
        """Check if content matches bias patterns"""
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in patterns)

    def _create_bias_assessment(self, bias_type: str, content: str) -> BiasAssessment:
        """Create bias assessment for detected bias"""
        description = self.i18n.get_text("bias_types", bias_type)
        potential_impact = self.i18n.get_text("potential_impacts", bias_type)
        mitigation_measures = self.i18n.get_text("bias_mitigation", bias_type)

        return BiasAssessment(
            bias_type=bias_type,
            severity="medium",  # Default medium severity
            description=description,
            potential_impact=potential_impact,
            mitigation_measures=mitigation_measures,
        )

    def _detect_methodology_biases(
        self, methodology: Dict[str, Any]
    ) -> List[BiasAssessment]:
        """Detect biases in research methodology"""
        biases = []

        # Detect sample selection bias
        if methodology.get("sampling_method") == "convenience":
            biases.append(
                BiasAssessment(
                    bias_type="selection_bias",
                    severity="high",
                    description=self.i18n.get_text("bias_types", "selection_bias"),
                    potential_impact=self.i18n.get_text(
                        "potential_impacts", "selection_bias"
                    ),
                    mitigation_measures=self.i18n.get_text(
                        "bias_mitigation", "selection_bias"
                    ),
                )
            )

        # Detect temporal bias
        if methodology.get("data_period") and "crisis" in methodology.get(
            "context", ""
        ):
            biases.append(
                BiasAssessment(
                    bias_type="temporal_bias",
                    severity="medium",
                    description=self.i18n.get_text("bias_types", "temporal_bias"),
                    potential_impact=self.i18n.get_text(
                        "potential_impacts", "temporal_bias"
                    ),
                    mitigation_measures=self.i18n.get_text(
                        "bias_mitigation", "temporal_bias"
                    ),
                )
            )

        return biases


class CriticalThinkingEngine:
    """Comprehensive Critical Thinking Analysis Engine"""

    def __init__(self, language: Language = Language.ZH_CN):
        self.language = language
        self.i18n = get_i18n_manager()
        self.i18n.set_language(language)

        self.limitation_detector = LimitationDetector(language)
        self.confidence_assessor = ConfidenceAssessor(language)
        self.bias_detector = BiasDetector(language)

    def set_language(self, language: Language):
        """Set language for the critical thinking engine"""
        self.language = language
        self.i18n.set_language(language)
        self.limitation_detector.set_language(language)
        self.confidence_assessor.set_language(language)
        self.bias_detector.set_language(language)

    def analyze_content(
        self, content: str, metadata: Dict[str, Any] = None
    ) -> CriticalAnalysis:
        """Perform comprehensive critical thinking analysis on content"""
        if metadata is None:
            metadata = {}

        analysis_id = str(uuid.uuid4())[:12]

        # Detect data limitations
        limitations = self.limitation_detector.detect_limitations(content, metadata)

        # Detect biases
        biases = self.bias_detector.detect_biases(
            content, metadata.get("methodology", {})
        )

        # Extract key statements and assess confidence
        key_statements = self._extract_key_statements(content)
        confidence_annotations = []

        for statement in key_statements:
            evidence = self._extract_supporting_evidence(statement, content)
            confidence = self.confidence_assessor.assess_confidence(statement, evidence)
            confidence_annotations.append(confidence)

        # Generate enhanced content
        enhanced_content = self._enhance_content_with_critical_thinking(
            content, limitations, biases, confidence_annotations
        )

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            limitations, biases, confidence_annotations
        )

        return CriticalAnalysis(
            analysis_id=analysis_id,
            original_content=content,
            confidence_annotations=confidence_annotations,
            identified_limitations=limitations,
            bias_assessments=biases,
            enhanced_content=enhanced_content,
            quality_score=quality_score,
        )

    def _extract_key_statements(self, content: str) -> List[str]:
        """Extract key statements from content"""
        # Language-specific sentence splitting and patterns
        if self.language == Language.ZH_CN:
            sentences = re.split(r"[。！？]", content)
            important_patterns = [
                r"结果显示",
                r"研究表明",
                r"数据证实",
                r"分析揭示",
                r"可以看出",
                r"表明",
                r"证明",
                r"显示",
            ]
        else:
            sentences = re.split(r"[.!?]", content)
            important_patterns = [
                r"results show",
                r"research indicates",
                r"data confirms",
                r"analysis reveals",
                r"can be seen",
                r"indicates",
                r"proves",
                r"demonstrates",
            ]

        key_statements = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in important_patterns
            ):
                if len(sentence) > 10:  # Filter out sentences that are too short
                    key_statements.append(sentence)

        return key_statements[:5]  # Return at most 5 key statements

    def _extract_supporting_evidence(
        self, statement: str, full_content: str
    ) -> List[str]:
        """Extract supporting evidence for statements"""
        evidence = []

        # Language-specific evidence patterns
        if self.language == Language.ZH_CN:
            patterns = [
                r"数据显示.*?[。！？]",  # Chinese: data shows
                r"根据.*?[。！？]",  # Chinese: according to
                r"研究发现.*?[。！？]",  # Chinese: research found
                r"\[.*?\]\(.*?\)",  # Markdown links
                r"来源[:：].*?[。！？]",  # Chinese: source
            ]
        else:
            patterns = [
                r"data shows.*?[.!?]",
                r"according to.*?[.!?]",
                r"research found.*?[.!?]",
                r"\[.*?\]\(.*?\)",  # Markdown links
                r"source[:：].*?[.!?]",
            ]

        for pattern in patterns:
            matches = re.findall(pattern, full_content, re.IGNORECASE)
            evidence.extend(matches[:2])  # At most 2 per type

        return evidence[:3]  # Return at most 3 evidence items

    def _enhance_content_with_critical_thinking(
        self,
        content: str,
        limitations: List[DataLimitation],
        biases: List[BiasAssessment],
        annotations: List[ConfidenceAnnotation],
    ) -> str:
        """Enhance content with critical thinking analysis"""
        enhanced = content + "\n\n"

        # Add data limitations explanation
        if limitations:
            enhanced += (
                f"## {self.i18n.get_text('section_titles', 'data_limitations')}\n\n"
            )
            for limitation in limitations:
                enhanced += f"**{limitation.limitation_type.value}**: {limitation.description}\n\n"
                if limitation.mitigation_suggestions:
                    enhanced += f"{self.i18n.get_text('section_titles', 'suggested_improvements')}:\n"
                    for suggestion in limitation.mitigation_suggestions:
                        enhanced += f"- {suggestion}\n"
                    enhanced += "\n"

        # Add bias alerts
        if biases:
            enhanced += f"## {self.i18n.get_text('section_titles', 'bias_alerts')}\n\n"
            for bias in biases:
                enhanced += f"**{bias.bias_type}**: {bias.description}\n\n"
                enhanced += f"{self.i18n.get_text('section_titles', 'potential_impact')}: {bias.potential_impact}\n\n"
                if bias.mitigation_measures:
                    enhanced += f"{self.i18n.get_text('section_titles', 'mitigation_measures')}:\n"
                    for measure in bias.mitigation_measures:
                        enhanced += f"- {measure}\n"
                    enhanced += "\n"

        # Add confidence annotations
        if annotations:
            enhanced += f"## {self.i18n.get_text('section_titles', 'conclusion_confidence')}\n\n"
            for annotation in annotations:
                enhanced += f"**{self.i18n.get_text('section_titles', 'statement')}**: {annotation.statement}\n\n"
                enhanced += f"**{self.i18n.get_text('section_titles', 'confidence')}**: {annotation.confidence_level.value}\n\n"
                enhanced += f"**{self.i18n.get_text('section_titles', 'reasoning')}**: {annotation.reasoning}\n\n"
                if annotation.supporting_evidence:
                    enhanced += f"**{self.i18n.get_text('section_titles', 'supporting_evidence')}**:\n"
                    for evidence in annotation.supporting_evidence:
                        enhanced += f"- {evidence}\n"
                    enhanced += "\n"
                enhanced += "---\n\n"

        return enhanced

    def _calculate_quality_score(
        self,
        limitations: List[DataLimitation],
        biases: List[BiasAssessment],
        annotations: List[ConfidenceAnnotation],
    ) -> float:
        """Calculate overall quality score based on analysis results"""
        base_score = 80.0  # Base score

        # Deduct points for limitations
        for limitation in limitations:
            if limitation.impact_level == "high":
                base_score -= 15
            elif limitation.impact_level == "medium":
                base_score -= 10
            else:
                base_score -= 5

        # Deduct points for biases
        for bias in biases:
            if bias.severity == "high":
                base_score -= 20
            elif bias.severity == "medium":
                base_score -= 15
            else:
                base_score -= 10

        # Add points for confidence
        high_confidence_count = sum(
            1
            for a in annotations
            if a.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        )
        base_score += high_confidence_count * 5

        return max(0.0, min(100.0, base_score))


# Usage example
def demo_critical_thinking(language: Language = Language.ZH_CN):
    """Demonstrate critical thinking engine functionality"""
    engine = CriticalThinkingEngine(language)

    # Sample content based on language
    if language == Language.ZH_CN:
        sample_content = """
        基于对100家公司的分析，研究表明科技股在2023年表现优异。
        数据显示平均收益率达到15%，远超市场预期。
        这一结果表明科技行业具有强劲的增长潜力。
        建议投资者增加科技股配置。
        """
    else:
        sample_content = """
        Based on analysis of 100 companies, research shows tech stocks performed excellently in 2023.
        Data shows average returns reached 15%, far exceeding market expectations.
        This result indicates the tech industry has strong growth potential.
        Recommend investors increase tech stock allocation.
        """

    # Sample metadata
    metadata = {
        "data_sources": ["wind"],
        "methodology": {
            "sampling_method": "convenience",
            "data_period": "2023",
            "context": "post_pandemic",
        },
        "has_control_group": False,
    }

    # Execute critical analysis
    analysis = engine.analyze_content(sample_content, metadata)

    print(f"Analysis ID: {analysis.analysis_id}")
    print(f"Quality score: {analysis.quality_score:.1f}")
    print(f"Limitations found: {len(analysis.identified_limitations)}")
    print(f"Biases found: {len(analysis.bias_assessments)}")
    print(f"Confidence annotations: {len(analysis.confidence_annotations)}")

    print("\nEnhanced content:")
    print(analysis.enhanced_content)


if __name__ == "__main__":
    demo_critical_thinking()
