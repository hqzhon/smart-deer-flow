"""Phase 2 Validation Script - Comprehensive Integration Testing

Phase 2 comprehensive validation script that integrates integration testing, compatibility verification, and performance analysis.
"""

import asyncio
import sys
import traceback
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import test modules
import pytest
from scripts.token_usage_analyzer import TokenUsageAnalyzer

# Import core components
from src.utils.context.execution_context_manager import ExecutionContextManager, ContextConfig
from src.utils.performance.parallel_executor import ParallelExecutor, ParallelTask, TaskPriority, SharedTaskContext
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_context_isolator import ResearcherContextConfig
from src.config.configuration import Configuration


class Phase2ValidationRunner:
    """Phase 2 validation runner"""
    
    def __init__(self):
        self.validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "phase": "Phase 2 - Integration Verification",
            "test_results": {},
            "performance_analysis": {},
            "compatibility_check": {},
            "overall_status": "PENDING",
            "recommendations": []
        }
        
        # Initialize components
        self.base_manager = ExecutionContextManager(ContextConfig())
        self.context_extension = ResearcherContextExtension(self.base_manager)
        self.token_analyzer = TokenUsageAnalyzer()
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("\n=== Running Integration Tests ===")
        
        try:
            # Run pytest tests
            test_file = "test_phase2_integration.py"
            
            # Check if test file exists
            if not Path(test_file).exists():
                return {
                    "status": "FAILED",
                    "error": f"Test file {test_file} does not exist",
                    "details": {}
                }
            
            print(f"Executing test file: {test_file}")
            
            # Use pytest.main to run tests
            exit_code = pytest.main([
                test_file,
                "-v",
                "--tb=short"
            ])
            
            # Read test results
            test_results = {"status": "PASSED" if exit_code == 0 else "FAILED"}
            
            try:
                with open("test_results.json", 'r') as f:
                    detailed_results = json.load(f)
                    test_results["details"] = detailed_results
            except FileNotFoundError:
                test_results["details"] = {"note": "Detailed test results file not generated"}
            
            print(f"Integration test results: {test_results['status']}")
            return test_results
            
        except Exception as e:
            error_result = {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"Integration test execution error: {e}")
            return error_result
    
    def run_compatibility_checks(self) -> Dict[str, Any]:
        """Run compatibility checks"""
        print("\n=== Running Compatibility Checks ===")
        
        compatibility_results = {
            "execution_context_manager": self._check_execution_context_manager_compatibility(),
            "parallel_executor": self._check_parallel_executor_compatibility(),
            "configuration_system": self._check_configuration_compatibility(),
            "error_handling": self._check_error_handling_compatibility()
        }
        
        # Calculate overall compatibility status
        all_passed = all(
            result.get("status") == "PASSED" 
            for result in compatibility_results.values()
        )
        
        compatibility_results["overall_status"] = "PASSED" if all_passed else "FAILED"
        
        print(f"Compatibility check results: {compatibility_results['overall_status']}")
        return compatibility_results
    
    def _check_execution_context_manager_compatibility(self) -> Dict[str, Any]:
        """Check ExecutionContextManager compatibility"""
        try:
            print("  Checking ExecutionContextManager compatibility...")
            
            # Test basic functionality
            test_steps = [
                {"step": "Test step 1", "execution_res": "Test result 1"},
                {"step": "Test step 2", "execution_res": "Test result 2"}
            ]
            current_step = {"step": "Current step", "description": "Test description"}
            
            # Test context preparation
            optimized_steps, context_info = self.base_manager.prepare_context_for_execution(
                test_steps, current_step, "researcher"
            )
            
            # Test observation management
            observations = ["Observation 1", "Observation 2", "Observation 3"]
            managed_obs = self.base_manager.manage_observations(observations[:-1], observations[-1])
            
            # Verify results
            assert isinstance(optimized_steps, list)
            assert isinstance(context_info, str)
            assert isinstance(managed_obs, list)
            assert len(managed_obs) == len(observations)
            
            return {
                "status": "PASSED",
                "details": {
                    "context_preparation": "Normal",
                    "observation_management": "Normal",
                    "optimized_steps_count": len(optimized_steps),
                    "managed_observations_count": len(managed_obs)
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _check_parallel_executor_compatibility(self) -> Dict[str, Any]:
        """Check ParallelExecutor compatibility"""
        try:
            print("  Checking ParallelExecutor compatibility...")
            
            # Create shared context and executor
            shared_context = SharedTaskContext()
            executor = ParallelExecutor(
                max_concurrent_tasks=2,
                enable_adaptive_scheduling=False,
                shared_context=shared_context
            )
            
            # Test task creation
            async def test_task(task_id: str):
                # Simple test task to verify basic functionality
                return f"Task {task_id} completed"
            
            # Create parallel tasks
            tasks = [
                ParallelTask(
                    task_id=f"compat_test_{i}",
                    func=test_task,
                    args=(f"task_{i}",),
                    priority=TaskPriority.NORMAL
                )
                for i in range(2)
            ]
            
            # Run async test
            async def run_parallel_test():
                # Add tasks to executor
                executor.add_tasks(tasks)
                results = await executor.execute_all()
                return results
            
            # Execute test
            results = asyncio.run(run_parallel_test())
            
            # Verify results
            assert len(results) == 2
            for task_id, result in results.items():
                assert result.status.value == "completed"
                assert isinstance(result.result, str)
            
            return {
                "status": "PASSED",
                "details": {
                    "parallel_execution": "Normal",
                    "task_isolation": "Normal",
                    "completed_tasks": len(results)
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _check_configuration_compatibility(self) -> Dict[str, Any]:
        """Check configuration system compatibility"""
        try:
            print("  Checking configuration system compatibility...")
            
            # Test configuration loading
            config = Configuration()
            
            # Test isolation configuration creation
            isolation_config = ResearcherContextConfig(
                isolation_level="moderate",
                max_context_steps=2,
                max_step_content_length=1500
            )
            
            # Verify configuration attributes
            assert hasattr(config, 'enable_parallel_execution')
            assert isolation_config.isolation_level == "moderate"
            assert isolation_config.max_context_steps == 2
            
            return {
                "status": "PASSED",
                "details": {
                    "configuration_loading": "Normal",
                    "isolation_config_creation": "Normal",
                    "config_attributes": "Complete"
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _check_error_handling_compatibility(self) -> Dict[str, Any]:
        """Check error handling compatibility"""
        try:
            print("  Checking error handling compatibility...")
            
            # Test invalid context ID handling
            result = self.context_extension.get_isolated_context("invalid_id")
            assert result is None
            
            # Test context cleanup
            config = ResearcherContextConfig(isolation_level="minimal")
            context_id = self.context_extension.create_isolated_context(
                "error_test", "Error Test", "Testing error handling", config
            )
            
            # Verify context exists
            assert context_id in self.context_extension.active_isolators
            
            # Test cleanup
            cleanup_result = self.context_extension.finalize_isolated_context(context_id)
            assert isinstance(cleanup_result, dict)
            assert context_id not in self.context_extension.active_isolators
            
            return {
                "status": "PASSED",
                "details": {
                    "invalid_context_handling": "Normal",
                    "context_cleanup": "Normal",
                    "error_recovery": "Normal"
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run performance analysis"""
        print("\n=== Running Performance Analysis ===")
        
        try:
            # Execute token usage analysis
            analysis_results = self.token_analyzer.run_comprehensive_analysis()
            
            # Save analysis report
            self.token_analyzer.save_analysis_report(
                analysis_results, 
                "phase2_token_analysis.json"
            )
            
            # Generate performance comparison report
            performance_report = self.token_analyzer.generate_performance_comparison()
            with open("phase2_performance_report.md", 'w', encoding='utf-8') as f:
                f.write(performance_report)
            
            # Extract key metrics
            summary = analysis_results["analysis_summary"]
            
            performance_results = {
                "status": "COMPLETED",
                "summary": summary,
                "key_metrics": {
                    "total_token_savings": summary["total_token_savings"],
                    "savings_ratio": summary["overall_savings_ratio"],
                    "average_compression": summary["average_compression_ratio"],
                    "scenarios_tested": summary["total_scenarios"]
                },
                "report_files": [
                    "phase2_token_analysis.json",
                    "phase2_performance_report.md"
                ]
            }
            
            print(f"Performance analysis completed, Token savings rate: {summary['overall_savings_ratio']:.1%}")
            return performance_results
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Generate recommendations based on test results
        test_status = self.validation_results["test_results"].get("status")
        if test_status == "FAILED":
            recommendations.append("Fix issues found in integration tests")
        
        # Generate recommendations based on compatibility checks
        compat_status = self.validation_results["compatibility_check"].get("overall_status")
        if compat_status == "FAILED":
            recommendations.append("Resolve compatibility issues to ensure seamless integration with existing architecture")
        
        # Generate recommendations based on performance analysis
        perf_analysis = self.validation_results["performance_analysis"]
        if perf_analysis.get("status") == "COMPLETED":
            savings_ratio = perf_analysis.get("key_metrics", {}).get("savings_ratio", 0)
            if savings_ratio > 0.5:
                recommendations.append("Token savings are significant, recommend enabling isolation features in production environment")
            elif savings_ratio > 0.3:
                recommendations.append("Token savings are good, recommend enabling isolation features in specific scenarios")
            else:
                recommendations.append("Token savings are limited, recommend optimizing isolation algorithms")
        
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "All tests passed, ready to proceed to Phase 3",
                "Recommend further testing under actual workloads",
                "Consider adding more monitoring metrics to track performance"
            ])
        
        return recommendations
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation"""
        print("Starting Phase 2 comprehensive validation...")
        print(f"Validation time: {self.validation_results['validation_timestamp']}")
        
        # 1. Run integration tests
        self.validation_results["test_results"] = self.run_integration_tests()
        
        # 2. Run compatibility checks
        self.validation_results["compatibility_check"] = self.run_compatibility_checks()
        
        # 3. Run performance analysis
        self.validation_results["performance_analysis"] = self.run_performance_analysis()
        
        # 4. Generate recommendations
        self.validation_results["recommendations"] = self.generate_recommendations()
        
        # 5. Determine overall status
        test_passed = self.validation_results["test_results"].get("status") in ["PASSED", "COMPLETED"]
        compat_passed = self.validation_results["compatibility_check"].get("overall_status") == "PASSED"
        perf_completed = self.validation_results["performance_analysis"].get("status") == "COMPLETED"
        
        if test_passed and compat_passed and perf_completed:
            self.validation_results["overall_status"] = "PASSED"
        elif perf_completed and (test_passed or compat_passed):
            self.validation_results["overall_status"] = "PARTIAL_PASS"
        else:
            self.validation_results["overall_status"] = "FAILED"
        
        return self.validation_results
    
    def save_validation_report(self, filename: str = "phase2_validation_report.json"):
        """Save validation report"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
        print(f"\nValidation report saved to: {filename}")
    
    def print_validation_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("Phase 2 Validation Summary")
        print("="*60)
        
        print(f"Overall status: {self.validation_results['overall_status']}")
        print(f"Validation time: {self.validation_results['validation_timestamp']}")
        
        print("\nDetailed results:")
        print(f"  Integration tests: {self.validation_results['test_results'].get('status', 'UNKNOWN')}")
        print(f"  Compatibility checks: {self.validation_results['compatibility_check'].get('overall_status', 'UNKNOWN')}")
        print(f"  Performance analysis: {self.validation_results['performance_analysis'].get('status', 'UNKNOWN')}")
        
        # Print performance metrics
        perf_metrics = self.validation_results['performance_analysis'].get('key_metrics', {})
        if perf_metrics:
            print("\nPerformance metrics:")
            print(f"  Token savings: {perf_metrics.get('total_token_savings', 0):,}")
            print(f"  Savings rate: {perf_metrics.get('savings_ratio', 0):.1%}")
            print(f"  Average compression: {perf_metrics.get('average_compression', 0):.1%}")
        
        # Print recommendations
        recommendations = self.validation_results.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """Main function"""
    try:
        # Create validation runner
        validator = Phase2ValidationRunner()
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Save report
        validator.save_validation_report()
        
        # Print summary
        validator.print_validation_summary()
        
        # Set exit code based on results
        if results["overall_status"] == "PASSED":
            print("\n✅ Phase 2 validation completed successfully!")
            sys.exit(0)
        elif results["overall_status"] == "PARTIAL_PASS":
            print("\n⚠️  Phase 2 validation partially passed, please check detailed report.")
            sys.exit(1)
        else:
            print("\n❌ Phase 2 validation failed, please check error messages and fix issues.")
            sys.exit(2)
            
    except Exception as e:
        print(f"\n💥 Error occurred during validation: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()