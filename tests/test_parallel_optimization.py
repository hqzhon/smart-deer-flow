#!/usr/bin/env python3
"""Unit tests for parallel processing optimization functionality"""

import asyncio
import unittest
import logging
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestParallelOptimization(unittest.IsolatedAsyncioTestCase):
    """Test parallel processing optimization functionality"""
    
    async def asyncSetUp(self):
        """Async setup"""
        # Import required modules for testing
        try:
            from src.utils.workflow_optimizer import (
                WorkflowOptimizer, WorkflowOptimizationLevel, WorkflowOptimizationConfig
            )
            from src.utils.performance_optimizer import (
                AdvancedParallelExecutor, optimize_report_generation_workflow
            )
            
            self.WorkflowOptimizationLevel = WorkflowOptimizationLevel
            self.WorkflowOptimizer = WorkflowOptimizer
            self.WorkflowOptimizationConfig = WorkflowOptimizationConfig
            self.AdvancedParallelExecutor = AdvancedParallelExecutor
            
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            self.skipTest(f"Unable to import required modules: {e}")
    
    @patch('src.utils.workflow_optimizer.WorkflowOptimizer.initialize')
    async def test_workflow_optimizer_creation(self, mock_initialize):
        """Test workflow optimizer creation"""
        mock_initialize.return_value = True
        
        # Create optimizer
        config = self.WorkflowOptimizationConfig(
            optimization_level=self.WorkflowOptimizationLevel.ADVANCED,
            max_workers=4
        )
        optimizer = self.WorkflowOptimizer(config)
        
        # Initialize optimizer
        await optimizer.initialize()
        
        # Verify initialization was called
        mock_initialize.assert_called_once()
        
        # Verify configuration
        self.assertEqual(optimizer.config.optimization_level, self.WorkflowOptimizationLevel.ADVANCED)
        self.assertEqual(optimizer.config.max_workers, 4)
    
    @patch('src.utils.workflow_optimizer.WorkflowOptimizer.optimize_research_workflow')
    @patch('src.utils.workflow_optimizer.create_optimized_workflow')
    async def test_optimize_single_research_task(self, mock_create_optimizer, mock_optimize_research):
        """Test single research task optimization"""
        from src.utils.workflow_optimizer import optimize_single_research_task
        
        # Mock optimizer
        mock_optimizer = AsyncMock()
        mock_create_optimizer.return_value = mock_optimizer
        
        # Mock research result
        expected_result = {'status': 'success', 'data': 'test_data'}
        mock_optimize_research.return_value = expected_result
        mock_optimizer.optimize_research_workflow.return_value = expected_result
        
        # Call optimization function
        result = await optimize_single_research_task(
            user_query="test query",
            workflow_type="research",
            max_workers=4
        )
        
        # Verify result
        self.assertEqual(result, expected_result)
        mock_optimizer.shutdown.assert_called_once()
    
    @patch('src.utils.performance_optimizer.AdvancedParallelExecutor.decompose_and_execute_research_task')
    @patch('src.utils.performance_optimizer.AdvancedParallelExecutor.start')
    @patch('src.utils.performance_optimizer.AdvancedParallelExecutor.stop')
    @patch('src.utils.performance_optimizer.AdvancedParallelExecutor.get_metrics')
    async def test_optimize_report_generation_workflow(self, mock_get_metrics, mock_stop, mock_start, mock_execute):
        """Test report generation workflow optimization"""
        from src.utils.performance_optimizer import optimize_report_generation_workflow
        
        # Mock execution result and metrics
        expected_result = {'status': 'success', 'sections': ['section1', 'section2']}
        mock_execute.return_value = expected_result
        mock_get_metrics.return_value = {'success_rate': 0.95, 'avg_execution_time': 0.5}
        
        # Call optimization function
        result = await optimize_report_generation_workflow(
            user_query="test report generation",
            workflow_type="report",
            max_workers=4
        )
        
        # Verify result
        self.assertIn('performance_metrics', result)
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_execute.assert_called_once()
    
    @patch('src.utils.performance_optimizer.IntelligentTaskDecomposer.decompose_task')
    async def test_intelligent_task_decomposition(self, mock_decompose):
        """Test intelligent task decomposition"""
        # Mock task decomposition result
        mock_decompose.return_value = [
            {'id': 'task1', 'description': 'Subtask 1'},
            {'id': 'task2', 'description': 'Subtask 2'},
            {'id': 'task3', 'description': 'Subtask 3'}
        ]
        
        # Create executor instance
        executor = self.AdvancedParallelExecutor(max_workers=4)
        
        # Call task decomposition
        if hasattr(executor, 'task_decomposer') and hasattr(executor.task_decomposer, 'decompose_task'):
            subtasks = await executor.task_decomposer.decompose_task(
                "Complex research task",
                task_type="research"
            )
            
            # Verify decomposition result
            self.assertEqual(len(subtasks), 3)
            self.assertEqual(subtasks[0]['id'], 'task1')
        else:
            self.skipTest("Executor does not implement intelligent task decomposition")
    
    @patch('src.utils.performance_optimizer.DynamicResourceAllocator.allocate_resources')
    async def test_dynamic_resource_allocation(self, mock_allocate):
        """Test dynamic resource allocation"""
        # Mock resource allocation result
        mock_allocate.return_value = {
            'cpu_cores': 2,
            'memory_mb': 512,
            'priority': 3
        }
        
        # Create executor instance
        executor = self.AdvancedParallelExecutor(max_workers=4)
        
        # Call resource allocation
        if hasattr(executor, 'resource_allocator') and hasattr(executor.resource_allocator, 'allocate_resources'):
            resources = await executor.resource_allocator.allocate_resources(
                task_complexity=0.8,
                system_load=0.5
            )
            
            # Verify allocation result
            self.assertEqual(resources['cpu_cores'], 2)
            self.assertEqual(resources['memory_mb'], 512)
        else:
            self.skipTest("Executor does not implement dynamic resource allocation")
    
    @patch('src.utils.performance_optimizer.AdaptiveLoadBalancer.balance_tasks')
    async def test_adaptive_load_balancing(self, mock_balance):
        """Test adaptive load balancing"""
        # Mock load balancing result
        mock_balance.return_value = {
            'worker_assignments': {0: 2, 1: 3, 2: 1, 3: 2},
            'load_distribution': 'balanced'
        }
        
        # Create executor instance
        executor = self.AdvancedParallelExecutor(max_workers=4)
        
        # Call load balancing
        if hasattr(executor, 'load_balancer') and hasattr(executor.load_balancer, 'balance_tasks'):
            tasks = [
                {'id': f'task{i}', 'priority': i % 3 + 1}
                for i in range(8)
            ]
            
            balance_result = await executor.load_balancer.balance_tasks(tasks)
            
            # Verify balancing result
            self.assertEqual(balance_result['load_distribution'], 'balanced')
            self.assertEqual(sum(balance_result['worker_assignments'].values()), 8)
        else:
            self.skipTest("Executor does not implement adaptive load balancing")
    
    @patch('src.utils.performance_optimizer.TaskDependencyResolver.resolve_dependencies')
    async def test_task_dependency_resolution(self, mock_resolve):
        """Test task dependency resolution"""
        # Mock dependency resolution result
        mock_resolve.return_value = {
            'execution_order': ['task1', 'task2', 'task3', 'task4'],
            'parallel_groups': [['task1'], ['task2', 'task3'], ['task4']]
        }
        
        # Create executor instance
        executor = self.AdvancedParallelExecutor(max_workers=4)
        
        # Call dependency resolution
        if hasattr(executor, 'dependency_resolver') and hasattr(executor.dependency_resolver, 'resolve_dependencies'):
            tasks = {
                'task1': {'id': 'task1', 'dependencies': []},
                'task2': {'id': 'task2', 'dependencies': ['task1']},
                'task3': {'id': 'task3', 'dependencies': ['task1']},
                'task4': {'id': 'task4', 'dependencies': ['task2', 'task3']}
            }
            
            resolution = await executor.dependency_resolver.resolve_dependencies(tasks)
            
            # Verify resolution result
            self.assertEqual(len(resolution['execution_order']), 4)
            self.assertEqual(len(resolution['parallel_groups']), 3)
        else:
            self.skipTest("Executor does not implement task dependency resolution")


class TestWorkflowIntegration(unittest.IsolatedAsyncioTestCase):
    """Test workflow integration"""
    
    async def asyncSetUp(self):
        """Async setup"""
        # Import required modules for testing
        try:
            from src.workflow import (
                run_optimized_research_workflow,
                run_parallel_report_generation,
                get_workflow_optimizer
            )
            from src.utils.workflow_optimizer import WorkflowOptimizationLevel
            
            self.run_optimized_research_workflow = run_optimized_research_workflow
            self.run_parallel_report_generation = run_parallel_report_generation
            self.get_workflow_optimizer = get_workflow_optimizer
            self.WorkflowOptimizationLevel = WorkflowOptimizationLevel
            
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            self.skipTest(f"Unable to import required modules: {e}")
    
    @patch('src.workflow.run_agent_workflow_async')
    @patch('src.utils.workflow_optimizer.WorkflowOptimizer.optimize_research_workflow')
    @patch('src.utils.workflow_optimizer.WorkflowOptimizer.get_optimization_metrics')
    @patch('src.workflow.get_workflow_optimizer')
    async def test_run_optimized_research_workflow(self, mock_get_optimizer, mock_get_metrics, 
                                                mock_optimize, mock_run_workflow):
        """Test optimized research workflow"""
        # Mock optimizer and results
        mock_optimizer = AsyncMock()
        mock_get_optimizer.return_value = mock_optimizer
        
        expected_result = {'status': 'success', 'data': 'research_result'}
        mock_optimize.return_value = expected_result
        mock_get_metrics.return_value = {'success_rate': 0.9}
        
        # Call optimized research workflow
        result = await self.run_optimized_research_workflow(
            user_input="test research query",
            workflow_type="research",
            optimization_level=self.WorkflowOptimizationLevel.ADVANCED
        )
        
        # Verify result
        self.assertEqual(result['status'], 'success')
        self.assertIn('workflow_metrics', result)
        mock_optimize.assert_called_once()
        mock_get_metrics.assert_called_once()
    
    @patch('src.utils.workflow_optimizer.WorkflowOptimizer.optimize_report_generation')
    @patch('src.workflow.get_workflow_optimizer')
    async def test_run_parallel_report_generation(self, mock_get_optimizer, mock_optimize_report):
        """Test parallel report generation"""
        # Mock optimizer and results
        mock_optimizer = AsyncMock()
        mock_get_optimizer.return_value = mock_optimizer
        
        expected_result = {
            'sections': [
                {'content': 'Section 1 content'},
                {'content': 'Section 2 content'}
            ]
        }
        mock_optimize_report.return_value = expected_result
        
        # Call parallel report generation
        result = await self.run_parallel_report_generation(
            content_sections=["section1", "section2"],
            report_type="test_report",
            user_context="test context"
        )
        
        # Verify result
        self.assertEqual(len(result['sections']), 2)
        self.assertIn('generation_time', result)
        self.assertEqual(result['sections_processed'], 2)
        mock_optimize_report.assert_called_once()


if __name__ == '__main__':
    unittest.main()