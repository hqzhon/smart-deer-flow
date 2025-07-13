"""Token Usage Analyzer - Phase 2 Performance Analysis

Analyzes and compares token usage before and after researcher node isolation,
providing quantitative performance metrics.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from src.utils.tokens.token_counter import count_tokens
from src.utils.context.execution_context_manager import ExecutionContextManager, ContextConfig
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_context_isolator import ResearcherContextConfig
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


@dataclass
class TokenUsageMetrics:
    """Token usage metrics"""
    total_tokens: int
    input_tokens: int
    output_tokens: int
    context_tokens: int
    message_tokens: int
    observation_tokens: int
    processing_time_ms: float
    memory_usage_bytes: int
    compression_ratio: float = 0.0
    
    def calculate_savings(self, baseline: 'TokenUsageMetrics') -> 'TokenSavings':
        """Calculate savings compared to baseline"""
        return TokenSavings(
            token_savings=baseline.total_tokens - self.total_tokens,
            token_savings_ratio=(baseline.total_tokens - self.total_tokens) / baseline.total_tokens if baseline.total_tokens > 0 else 0,
            time_savings_ms=baseline.processing_time_ms - self.processing_time_ms,
            memory_savings_bytes=baseline.memory_usage_bytes - self.memory_usage_bytes,
            compression_improvement=self.compression_ratio - baseline.compression_ratio
        )


@dataclass
class TokenSavings:
    """Token savings metrics"""
    token_savings: int
    token_savings_ratio: float
    time_savings_ms: float
    memory_savings_bytes: int
    compression_improvement: float
    
    def to_summary(self) -> str:
        """Generate savings summary"""
        return f"""Token Savings Analysis:
- Token savings: {self.token_savings:,} ({self.token_savings_ratio:.1%})
- Time savings: {self.time_savings_ms:.1f}ms
- Memory savings: {self.memory_savings_bytes:,} bytes
- Compression improvement: {self.compression_improvement:.1%}"""


@dataclass
class AnalysisScenario:
    """Analysis scenario configuration"""
    name: str
    description: str
    messages: List[BaseMessage]
    observations: List[str]
    completed_steps: List[Dict[str, Any]]
    isolation_config: Optional[ResearcherContextConfig] = None


class TokenUsageAnalyzer:
    """Token usage analyzer"""
    
    def __init__(self):
        self.base_manager = ExecutionContextManager(ContextConfig(
            max_context_steps=5,
            max_step_content_length=2000,
            max_observations_length=4000,
            enable_content_deduplication=True,
            enable_smart_truncation=True
        ))
        self.context_extension = ResearcherContextExtension(self.base_manager)
        self.analysis_results: List[Dict[str, Any]] = []
    
    def create_test_scenarios(self) -> List[AnalysisScenario]:
        """Create test scenarios"""
        scenarios = []
        
        # Scenario 1: Lightweight research task
        scenarios.append(AnalysisScenario(
            name="lightweight_research",
            description="Lightweight research task - small amount of data",
            messages=[
                HumanMessage(content="Please research the basic concepts of artificial intelligence"),
                AIMessage(content="I will research the basic concepts of artificial intelligence, including definitions, history, and application areas."),
                HumanMessage(content="Focus on the machine learning part")
            ],
            observations=[
                "Artificial intelligence is a branch of computer science",
                "Machine learning is one of the core technologies of AI",
                "Deep learning is a subset of machine learning"
            ],
            completed_steps=[
                {"step": "concept_definition", "execution_res": "Completed the definition and classification of basic AI concepts"},
                {"step": "history_review", "execution_res": "Reviewed the major historical milestones in AI development"}
            ]
        ))
        
        # Scenario 2: Medium complexity research task
        scenarios.append(AnalysisScenario(
            name="medium_research",
            description="Medium complexity research task - moderate amount of data",
            messages=[
                HumanMessage(content="Please conduct in-depth research on the technical principles and application prospects of large language models"),
                AIMessage(content="I will conduct in-depth research on large language models from multiple dimensions including technical architecture, training methods, and application scenarios." + " Detailed analysis " * 50),
                HumanMessage(content="Pay special attention to the innovations of the Transformer architecture"),
                AIMessage(content="The core innovations of the Transformer architecture include self-attention mechanisms, positional encoding, etc." + " Technical details " * 100),
                HumanMessage(content="Compare the performance of different models")
            ],
            observations=[
                "The Transformer architecture has revolutionarily changed the NLP field" + " Detailed description " * 30,
                "The GPT series models have demonstrated powerful generation capabilities" + " Performance analysis " * 40,
                "BERT models perform excellently in understanding tasks" + " Technical comparison " * 35,
                "Multimodal large models are developing rapidly" + " Trend analysis " * 25,
                "Model scale and performance show scaling law relationships" + " Data support " * 45
            ],
            completed_steps=[
                {"step": "architecture_analysis", "execution_res": "In-depth analysis of the technical details and innovations of the Transformer architecture" + " Detailed content " * 60},
                {"step": "model_comparison", "execution_res": "Compared the characteristics and performance of mainstream models like GPT, BERT, T5" + " Comparison analysis " * 80},
                {"step": "application_research", "execution_res": "Researched application cases and effects of large language models in various fields" + " Application cases " * 70}
            ]
        ))
        
        # Scenario 3: Heavy research task
        scenarios.append(AnalysisScenario(
            name="heavy_research",
            description="Heavy research task - large amount of data",
            messages=[
                HumanMessage(content="Please comprehensively research the current status, technical challenges, and development trends of artificial intelligence applications in the healthcare field"),
                AIMessage(content="I will comprehensively research AI applications in healthcare from multiple dimensions including medical imaging, drug discovery, clinical diagnosis, and personalized treatment." + " Research plan " * 100),
                HumanMessage(content="Focus on analyzing breakthroughs of deep learning in medical image diagnosis"),
                AIMessage(content="Deep learning has achieved significant breakthroughs in medical image diagnosis, particularly in lung cancer screening and fundus lesion detection." + " Technical breakthroughs " * 150),
                HumanMessage(content="Analyze the latest progress in AI-assisted drug discovery"),
                AIMessage(content="AI applications in drug discovery include molecular design, drug screening, and clinical trial optimization." + " Drug discovery " * 120),
                HumanMessage(content="Evaluate the impact of regulatory policies on AI medical applications")
            ],
            observations=[
                "FDA has approved multiple AI medical devices for market" + " Regulatory progress " * 80,
                "Medical imaging AI accuracy has reached expert level" + " Performance evaluation " * 90,
                "AI drug discovery can significantly shorten R&D cycles" + " Efficiency improvement " * 70,
                "Personalized medicine is becoming a new treatment paradigm" + " Medical transformation " * 85,
                "Data privacy and security are key challenges for AI healthcare" + " Challenge analysis " * 75,
                "Multimodal AI shows potential in comprehensive diagnosis" + " Technology integration " * 65,
                "Explainability requirements for medical AI are increasingly important" + " Explainability " * 55,
                "Cross-institutional data sharing faces technical and policy barriers" + " Data sharing " * 95
            ],
            completed_steps=[
                {"step": "status_research", "execution_res": "Comprehensively researched the current status and achievements of AI applications in various medical sub-fields" + " Research report " * 120},
                {"step": "technical_analysis", "execution_res": "In-depth analysis of applications of deep learning, computer vision, NLP and other technologies in healthcare" + " Technical depth " * 140},
                {"step": "case_study", "execution_res": "Studied medical AI products and solutions from companies like Google, IBM, and Baidu" + " Case analysis " * 110},
                {"step": "challenge_identification", "execution_res": "Identified key challenges such as data quality, algorithmic bias, and regulatory compliance" + " Challenge analysis " * 100},
                {"step": "trend_prediction", "execution_res": "Predicted development trends and future opportunities in AI healthcare" + " Trend analysis " * 130}
            ]
        ))
        
        return scenarios
    
    def measure_standard_processing(self, scenario: AnalysisScenario) -> TokenUsageMetrics:
        """Measure token usage of standard processing approach"""
        start_time = time.time()
        
        # Calculate message tokens
        message_tokens = sum(
            count_tokens(msg.content).total_tokens for msg in scenario.messages
        )
        
        # Calculate observation tokens
        observation_tokens = sum(
            count_tokens(obs).total_tokens for obs in scenario.observations
        )
        
        # Calculate step tokens
        step_tokens = sum(
            count_tokens(step.get("execution_res", "")).total_tokens 
            for step in scenario.completed_steps
        )
        
        # Use base manager to process context
        optimized_steps, context_info = self.base_manager.prepare_context_for_execution(
            scenario.completed_steps, 
            {"step": "current_step", "description": scenario.description},
            "researcher"
        )
        
        # Manage observations
        if scenario.observations:
            managed_observations = self.base_manager.manage_observations(
                scenario.observations[:-1], scenario.observations[-1]
            )
        else:
            managed_observations = []
        
        # Optimize messages
        optimized_messages = self.base_manager.optimize_messages(
            scenario.messages, token_limit=4000
        )
        
        # Calculate processed tokens
        processed_message_tokens = sum(
            count_tokens(msg.content).total_tokens for msg in optimized_messages
        )
        processed_observation_tokens = sum(
            count_tokens(obs).total_tokens for obs in managed_observations
        )
        processed_context_tokens = count_tokens(context_info).total_tokens
        
        processing_time = (time.time() - start_time) * 1000
        
        total_original = message_tokens + observation_tokens + step_tokens
        total_processed = processed_message_tokens + processed_observation_tokens + processed_context_tokens
        
        return TokenUsageMetrics(
            total_tokens=total_processed,
            input_tokens=total_original,
            output_tokens=total_processed,
            context_tokens=processed_context_tokens,
            message_tokens=processed_message_tokens,
            observation_tokens=processed_observation_tokens,
            processing_time_ms=processing_time,
            memory_usage_bytes=len(str(scenario.messages + scenario.observations + scenario.completed_steps)),
            compression_ratio=(total_original - total_processed) / total_original if total_original > 0 else 0
        )
    
    def measure_isolated_processing(self, scenario: AnalysisScenario) -> TokenUsageMetrics:
        """Measure token usage of isolated processing approach"""
        start_time = time.time()
        
        # Create isolated context
        config = scenario.isolation_config or ResearcherContextConfig(
            isolation_level="moderate",
            max_context_steps=2,
            max_step_content_length=2000
        )
        
        context_id = self.context_extension.create_isolated_context(
            f"analysis_{scenario.name}",
            scenario.description,
            f"Analysis scenario: {scenario.name}",
            config
        )
        
        isolator = self.context_extension.active_isolators[context_id]
        
        # Simulate adding data to isolated context
        for msg in scenario.messages:
            isolator.capture_local_activity("message", msg.content)
        
        for obs in scenario.observations:
            isolator.capture_local_activity("search", {"observation": obs})
        
        for step in scenario.completed_steps:
            isolator.capture_local_activity("analysis", step.get("execution_res", ""))
        
        # Generate refined output
        refined_output = isolator.generate_refined_output()
        
        # Calculate token usage
        output_tokens = count_tokens(refined_output).total_tokens
        context_size = isolator.estimate_context_size()
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate original data tokens
        original_tokens = sum(
            count_tokens(msg.content).total_tokens for msg in scenario.messages
        ) + sum(
            count_tokens(obs).total_tokens for obs in scenario.observations
        ) + sum(
            count_tokens(step.get("execution_res", "")).total_tokens 
            for step in scenario.completed_steps
        )
        
        # Clean up context
        self.context_extension.finalize_isolated_context(context_id)
        
        return TokenUsageMetrics(
            total_tokens=output_tokens,
            input_tokens=original_tokens,
            output_tokens=output_tokens,
            context_tokens=context_size,
            message_tokens=0,  # Compressed into refined_output
            observation_tokens=0,  # Compressed into refined_output
            processing_time_ms=processing_time,
            memory_usage_bytes=len(refined_output),
            compression_ratio=(original_tokens - output_tokens) / original_tokens if original_tokens > 0 else 0
        )
    
    def analyze_scenario(self, scenario: AnalysisScenario) -> Dict[str, Any]:
        """Analyze a single scenario"""
        print(f"\nAnalyzing scenario: {scenario.name} - {scenario.description}")
        
        # Measure standard processing
        print("  Measuring standard processing approach...")
        standard_metrics = self.measure_standard_processing(scenario)
        
        # Measure isolated processing
        print("  Measuring isolated processing approach...")
        isolated_metrics = self.measure_isolated_processing(scenario)
        
        # Calculate savings
        savings = isolated_metrics.calculate_savings(standard_metrics)
        
        result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "standard_metrics": asdict(standard_metrics),
            "isolated_metrics": asdict(isolated_metrics),
            "savings": asdict(savings),
            "analysis_time": datetime.now().isoformat()
        }
        
        # Print result summary
        print(f"  Standard processing: {standard_metrics.total_tokens:,} tokens")
        print(f"  Isolated processing: {isolated_metrics.total_tokens:,} tokens")
        print(f"  Savings: {savings.token_savings:,} tokens ({savings.token_savings_ratio:.1%})")
        
        return result
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        print("Starting comprehensive token usage analysis...")
        
        scenarios = self.create_test_scenarios()
        results = []
        
        for scenario in scenarios:
            result = self.analyze_scenario(scenario)
            results.append(result)
            self.analysis_results.append(result)
        
        # Calculate overall statistics
        total_standard_tokens = sum(r["standard_metrics"]["total_tokens"] for r in results)
        total_isolated_tokens = sum(r["isolated_metrics"]["total_tokens"] for r in results)
        total_savings = total_standard_tokens - total_isolated_tokens
        total_savings_ratio = total_savings / total_standard_tokens if total_standard_tokens > 0 else 0
        
        summary = {
            "analysis_summary": {
                "total_scenarios": len(scenarios),
                "total_standard_tokens": total_standard_tokens,
                "total_isolated_tokens": total_isolated_tokens,
                "total_token_savings": total_savings,
                "overall_savings_ratio": total_savings_ratio,
                "average_compression_ratio": sum(r["isolated_metrics"]["compression_ratio"] for r in results) / len(results),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "scenario_results": results
        }
        
        print(f"\n=== Comprehensive Analysis Results ===")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Standard processing total tokens: {total_standard_tokens:,}")
        print(f"Isolated processing total tokens: {total_isolated_tokens:,}")
        print(f"Total token savings: {total_savings:,} ({total_savings_ratio:.1%})")
        print(f"Average compression ratio: {summary['analysis_summary']['average_compression_ratio']:.1%}")
        
        return summary
    
    def save_analysis_report(self, analysis_results: Dict[str, Any], filename: str = "token_analysis_report.json"):
        """Save analysis report"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        print(f"\nAnalysis report saved to: {filename}")
    
    def generate_performance_comparison(self) -> str:
        """Generate performance comparison report"""
        if not self.analysis_results:
            return "No analysis results available"
        
        report = ["# Token Usage Performance Comparison Report\n"]
        
        for result in self.analysis_results:
            scenario = result["scenario"]
            desc = result["description"]
            standard = result["standard_metrics"]
            isolated = result["isolated_metrics"]
            savings = result["savings"]
            
            report.append(f"## {scenario} - {desc}\n")
            report.append(f"- **Standard processing**: {standard['total_tokens']:,} tokens")
            report.append(f"- **Isolated processing**: {isolated['total_tokens']:,} tokens")
            report.append(f"- **Token savings**: {savings['token_savings']:,} ({savings['token_savings_ratio']:.1%})")
            report.append(f"- **Processing time**: Standard {standard['processing_time_ms']:.1f}ms vs Isolated {isolated['processing_time_ms']:.1f}ms")
            report.append(f"- **Compression ratio**: {isolated['compression_ratio']:.1%}\n")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Run token usage analysis
    analyzer = TokenUsageAnalyzer()
    
    # Execute comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Save report
    analyzer.save_analysis_report(results)
    
    # Generate performance comparison report
    performance_report = analyzer.generate_performance_comparison()
    with open("performance_comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(performance_report)
    
    print("\nAnalysis completed! Please check the generated report files.")