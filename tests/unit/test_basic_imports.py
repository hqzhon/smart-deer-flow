"""Unit tests for basic module imports."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class TestBasicImports:
    """Test cases for basic module imports."""
    
    def test_config_loader_import(self):
        """Test ConfigLoader import."""
        from src.config.config_loader import ConfigLoader
        assert ConfigLoader is not None
    
    def test_workflow_import(self):
        """Test workflow module import."""
        from src.workflow import run_agent_workflow_async
        assert run_agent_workflow_async is not None
    
    def test_workflow_optimizer_import(self):
        """Test WorkflowOptimizer import."""
        from src.utils.performance.workflow_optimizer import WorkflowOptimizer
        assert WorkflowOptimizer is not None
    
    def test_memory_manager_import(self):
        """Test HierarchicalMemoryManager import."""
        from src.utils.performance.memory_manager import HierarchicalMemoryManager
        assert HierarchicalMemoryManager is not None
    
    def test_content_processor_import(self):
        """Test ContentProcessor import."""
        from src.utils.tokens.content_processor import ContentProcessor
        assert ContentProcessor is not None
    
    def test_context_evaluator_import(self):
        """Test ContextStateEvaluator import."""
        from src.utils.context.context_evaluator import ContextStateEvaluator
        assert ContextStateEvaluator is not None
    
    def test_execution_context_manager_import(self):
        """Test ExecutionContextManager import."""
        from src.utils.context.execution_context_manager import ExecutionContextManager
        assert ExecutionContextManager is not None
    
    def test_error_handler_import(self):
        """Test LLMErrorHandler import."""
        from src.llms.error_handler import LLMErrorHandler
        assert LLMErrorHandler is not None
    
    def test_all_core_imports(self):
        """Test all core module imports together."""
        # Import all modules in one test to verify no conflicts
        from src.config.config_loader import ConfigLoader
        from src.workflow import run_agent_workflow_async
        from src.utils.performance.workflow_optimizer import WorkflowOptimizer
        from src.utils.performance.memory_manager import HierarchicalMemoryManager
        from src.utils.tokens.content_processor import ContentProcessor
        from src.utils.context.context_evaluator import ContextStateEvaluator
        from src.utils.context.execution_context_manager import ExecutionContextManager
        from src.llms.error_handler import LLMErrorHandler
        
        # Verify all imports are successful
        assert all([
            ConfigLoader,
            run_agent_workflow_async,
            WorkflowOptimizer,
            HierarchicalMemoryManager,
            ContentProcessor,
            ContextStateEvaluator,
            ExecutionContextManager,
            LLMErrorHandler
        ])
    
    def test_import_error_handling(self):
        """Test handling of import errors for non-existent modules."""
        with pytest.raises(ImportError):
            from src.non_existent_module import NonExistentClass
    
    def test_module_instantiation(self):
        """Test that imported modules can be instantiated."""
        from src.config.config_loader import ConfigLoader
        from src.utils.performance.workflow_optimizer import WorkflowOptimizer
        from src.utils.performance.memory_manager import HierarchicalMemoryManager
        from src.utils.tokens.content_processor import ContentProcessor
        from src.utils.context.context_evaluator import ContextStateEvaluator
        from src.utils.context.execution_context_manager import ExecutionContextManager
        from src.llms.error_handler import LLMErrorHandler
        
        # Test instantiation
        config_loader = ConfigLoader()
        optimizer = WorkflowOptimizer()
        memory_manager = HierarchicalMemoryManager()
        processor = ContentProcessor()
        evaluator = ContextStateEvaluator()
        context_manager = ExecutionContextManager()
        error_handler = LLMErrorHandler()
        
        # Verify instances are created
        assert isinstance(config_loader, ConfigLoader)
        assert isinstance(optimizer, WorkflowOptimizer)
        assert isinstance(memory_manager, HierarchicalMemoryManager)
        assert isinstance(processor, ContentProcessor)
        assert isinstance(evaluator, ContextStateEvaluator)
        assert isinstance(context_manager, ExecutionContextManager)
        assert isinstance(error_handler, LLMErrorHandler)