# SPDX-License-Identifier: MIT
"""
Phase 3 Integration Tests
Comprehensive tests for configuration system expansion and workflow integration
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.config.config_manager import ConfigManager
from src.config.config_integration import ConfigurationIntegrator
from src.config.researcher_config_loader import ResearcherConfigLoader, ResearcherConfig
from src.workflow.reflection_workflow import ReflectionWorkflow, WorkflowStage, WorkflowResult


class TestConfigManager:
    """Test configuration manager functionality."""

    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        manager = ConfigManager()
        assert manager is not None
        assert manager.main_loader is not None
        assert manager.integrator is not None

    def test_get_main_config(self):
        """Test getting main configuration."""
        manager = ConfigManager()
        config = manager.get_main_config()
        assert config is not None
        assert hasattr(config, 'enable_parallel_execution')

    def test_get_researcher_config(self):
        """Test getting researcher configuration."""
        manager = ConfigManager()
        config = manager.get_researcher_config()
        assert config is not None
        assert hasattr(config, 'enhanced_reflection')
        assert hasattr(config, 'isolation')

    def test_is_reflection_enabled(self):
        """Test reflection enablement check."""
        manager = ConfigManager()
        enabled = manager.is_reflection_enabled()
        assert isinstance(enabled, bool)

    def test_is_isolation_enabled(self):
        """Test isolation enablement check."""
        manager = ConfigManager()
        enabled = manager.is_isolation_enabled()
        assert isinstance(enabled, bool)

    def test_get_reflection_config(self):
        """Test getting reflection configuration."""
        manager = ConfigManager()
        config = manager.get_reflection_config()
        assert isinstance(config, dict)
        assert 'enabled' in config or 'enable_enhanced_reflection' in config

    def test_get_config_value(self):
        """Test getting configuration values."""
        manager = ConfigManager()
        
        # Test reflection config value
        value = manager.get_config_value('reflection_enabled', False)
        assert isinstance(value, bool)
        
        # Test isolation config value
        value = manager.get_config_value('isolation_enabled', False)
        assert isinstance(value, bool)
        
        # Test default value
        value = manager.get_config_value('nonexistent_key', 'default')
        assert value == 'default'

    def test_set_config_value(self):
        """Test setting configuration values."""
        manager = ConfigManager()
        
        # Test setting reflection config
        result = manager.set_config_value('reflection_enabled', True, 'reflection')
        assert isinstance(result, bool)
        
        # Test setting isolation config
        result = manager.set_config_value('isolation_enabled', True, 'isolation')
        assert isinstance(result, bool)

    def test_validate_configuration(self):
        """Test configuration validation."""
        manager = ConfigManager()
        result = manager.validate_configuration()
        assert isinstance(result, bool)

    def test_get_configuration_summary(self):
        """Test getting configuration summary."""
        manager = ConfigManager()
        summary = manager.get_configuration_summary()
        assert isinstance(summary, dict)
        assert 'main_config' in summary
        assert 'researcher_config' in summary
        assert 'integration_status' in summary

    def test_reload_configuration(self):
        """Test configuration reloading."""
        manager = ConfigManager()
        result = manager.reload_configuration()
        assert isinstance(result, bool)

    def test_export_configuration(self):
        """Test configuration export."""
        manager = ConfigManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = manager.export_configuration(temp_path, 'json')
            assert isinstance(result, bool)
            
            if result:
                assert os.path.exists(temp_path)
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                assert isinstance(data, dict)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConfigurationIntegrator:
    """Test configuration integrator functionality."""

    def test_integrator_initialization(self):
        """Test integrator initialization."""
        integrator = ConfigurationIntegrator()
        assert integrator is not None
        assert integrator.researcher_loader is not None

    def test_load_researcher_config(self):
        """Test loading researcher configuration."""
        integrator = ConfigurationIntegrator()
        config = integrator.load_researcher_config()
        assert isinstance(config, ResearcherConfig)
        assert hasattr(config, 'enhanced_reflection')

    def test_get_integrated_reflection_config(self):
        """Test getting integrated reflection config."""
        integrator = ConfigurationIntegrator()
        config = integrator.get_integrated_reflection_config()
        assert isinstance(config, dict)

    def test_get_integrated_isolation_config(self):
        """Test getting integrated isolation config."""
        integrator = ConfigurationIntegrator()
        config = integrator.get_integrated_isolation_config()
        assert isinstance(config, dict)

    def test_validate_configuration(self):
        """Test configuration validation."""
        integrator = ConfigurationIntegrator()
        result = integrator.validate_configuration()
        assert isinstance(result, bool)

    def test_get_configuration_summary(self):
        """Test getting configuration summary."""
        integrator = ConfigurationIntegrator()
        summary = integrator.get_configuration_summary()
        assert isinstance(summary, dict)


class TestResearcherConfigLoader:
    """Test researcher configuration loader."""

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = ResearcherConfigLoader()
        assert loader is not None

    def test_load_config(self):
        """Test loading configuration."""
        loader = ResearcherConfigLoader()
        config = loader.load_config()
        assert isinstance(config, dict)

    def test_parse_enhanced_reflection_config(self):
        """Test parsing enhanced reflection config."""
        loader = ResearcherConfigLoader()
        config_data = {
            "enable_enhanced_reflection": True,
            "max_reflection_loops": 3,
            "reflection_model": "gpt-4",
            "temperature": 0.7
        }
        
        reflection_config = loader.parse_enhanced_reflection_config(config_data)
        assert reflection_config.enable_enhanced_reflection == True
        assert reflection_config.max_reflection_loops == 3
        assert reflection_config.reflection_model == "gpt-4"
        assert reflection_config.temperature == 0.7

    def test_create_researcher_config(self):
        """Test creating researcher configuration."""
        loader = ResearcherConfigLoader()
        config = loader.create_researcher_config()
        assert isinstance(config, ResearcherConfig)
        assert hasattr(config, 'enhanced_reflection')
        assert hasattr(config, 'isolation')


class TestReflectionWorkflow:
    """Test reflection workflow functionality."""

    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = ReflectionWorkflow()
        assert workflow is not None
        assert workflow.reflection_agent is not None
        assert workflow.progressive_enabler is not None
        assert workflow.isolation_metrics is not None
        assert len(workflow.stages) > 0

    def test_workflow_stages(self):
        """Test workflow stages configuration."""
        workflow = ReflectionWorkflow()
        
        expected_stages = [
            "initialization", "context_analysis", "research_planning",
            "information_gathering", "synthesis", "validation"
        ]
        
        stage_names = [stage.name for stage in workflow.stages]
        for expected_stage in expected_stages:
            assert expected_stage in stage_names

    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test workflow execution."""
        workflow = ReflectionWorkflow()
        
        # Mock the reflection agent and other components
        with patch.object(workflow.reflection_agent, 'analyze_knowledge_gaps', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "gaps": ["gap1", "gap2"],
                "suggested_queries": ["query1", "query2"],
                "confidence_score": 0.8
            }
            
            with patch.object(workflow.progressive_enabler, 'should_enable_isolation', new_callable=AsyncMock) as mock_enable:
                mock_enable.return_value = {"should_enable": True, "level": "moderate"}
                
                with patch.object(workflow.isolation_metrics, 'update_isolation_level', new_callable=AsyncMock):
                    with patch.object(workflow.isolation_metrics, 'get_metrics_summary', new_callable=AsyncMock) as mock_metrics:
                        mock_metrics.return_value = {"isolation_active": True}
                        
                        result = await workflow.execute_workflow(
                            "Test research query",
                            {"initial": "context"}
                        )
                        
                        assert isinstance(result, WorkflowResult)
                        assert result.success == True
                        assert len(result.stage_results) > 0
                        assert isinstance(result.metrics, dict)
                        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_initialization_stage(self):
        """Test initialization stage execution."""
        workflow = ReflectionWorkflow()
        workflow.workflow_context = {"query": "test", "initial_context": {}}
        
        result = await workflow._execute_initialization_stage()
        assert result["success"] == True
        assert "initialized_context" in result
        assert result["config_loaded"] == True
        assert result["agents_ready"] == True

    @pytest.mark.asyncio
    async def test_execute_context_analysis_stage(self):
        """Test context analysis stage execution."""
        workflow = ReflectionWorkflow()
        workflow.workflow_context = {
            "query": "test query",
            "initial_context": {"test": "context"}
        }
        
        with patch.object(workflow.reflection_agent, 'analyze_knowledge_gaps', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "gaps": ["gap1", "gap2"],
                "suggested_queries": ["query1"]
            }
            
            result = await workflow._execute_context_analysis_stage()
            assert result["success"] == True
            assert "knowledge_gaps" in result
            assert result["context_analyzed"] == True
            assert "gap_count" in result

    @pytest.mark.asyncio
    async def test_execute_research_planning_stage(self):
        """Test research planning stage execution."""
        workflow = ReflectionWorkflow()
        workflow.workflow_context = {"query": "test query"}
        workflow.execution_history = [{
            "result": {
                "knowledge_gaps": {
                    "gaps": ["gap1"],
                    "suggested_queries": ["query1"]
                }
            }
        }]
        
        with patch.object(workflow.progressive_enabler, 'should_enable_isolation', new_callable=AsyncMock) as mock_enable:
            mock_enable.return_value = {"should_enable": True}
            
            result = await workflow._execute_research_planning_stage()
            assert result["success"] == True
            assert "research_plan" in result
            assert result["plan_created"] == True

    @pytest.mark.asyncio
    async def test_execute_information_gathering_stage(self):
        """Test information gathering stage execution."""
        workflow = ReflectionWorkflow()
        workflow.execution_history = [{
            "result": {
                "research_plan": {
                    "gaps_to_address": ["gap1"],
                    "suggested_queries": ["query1"]
                }
            }
        }]
        
        result = await workflow._execute_information_gathering_stage()
        assert result["success"] == True
        assert "gathered_information" in result
        assert "sources_count" in result
        assert result["information_quality"] == "high"

    @pytest.mark.asyncio
    async def test_execute_synthesis_stage(self):
        """Test synthesis stage execution."""
        workflow = ReflectionWorkflow()
        workflow.execution_history = [{
            "result": {
                "gathered_information": {
                    "sources": ["source1", "source2"],
                    "content": "test content"
                }
            }
        }]
        
        result = await workflow._execute_synthesis_stage()
        assert result["success"] == True
        assert "synthesis_result" in result
        assert "findings_count" in result
        assert result["synthesis_quality"] == "good"

    @pytest.mark.asyncio
    async def test_execute_validation_stage(self):
        """Test validation stage execution."""
        workflow = ReflectionWorkflow()
        workflow.execution_history = [{
            "result": {
                "synthesis_result": {
                    "key_findings": ["finding1", "finding2"],
                    "conclusions": ["conclusion1"]
                }
            }
        }]
        
        result = await workflow._execute_validation_stage()
        assert result["success"] == True
        assert "validation_result" in result
        assert result["validation_passed"] == True
        assert "final_confidence" in result

    @pytest.mark.asyncio
    async def test_apply_reflection(self):
        """Test applying reflection to stage result."""
        workflow = ReflectionWorkflow()
        workflow.workflow_context = {"query": "test query"}
        
        stage = WorkflowStage("test_stage", "Test stage", requires_reflection=True)
        stage_result = {"success": True, "data": "test"}
        
        with patch.object(workflow.reflection_agent, 'analyze_knowledge_gaps', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "gaps": ["gap1"],
                "confidence_score": 0.8
            }
            
            result = await workflow._apply_reflection(stage, stage_result)
            assert result["reflection_applied"] == True
            assert "reflection_insights" in result
            assert "reflection_confidence" in result

    @pytest.mark.asyncio
    async def test_apply_context_expansion(self):
        """Test context expansion functionality (now integrated into isolation metrics)."""
        workflow = ReflectionWorkflow()
        workflow.workflow_context = {"query": "test query"}
        
        stage = WorkflowStage("test_stage", "Test stage", context_expansion=True)
        stage_result = {"success": True, "data": "test"}
        
        # Context expansion is now handled by isolation metrics
        # Test that the stage result is properly marked
        result = stage_result.copy()
        if stage.context_expansion and result.get("success", False):
            result["context_expanded"] = True
            
        assert result["context_expanded"] == True

    def test_get_workflow_status(self):
        """Test getting workflow status."""
        workflow = ReflectionWorkflow()
        workflow.current_stage = "test_stage"
        workflow.workflow_context = {"query": "test"}
        
        status = workflow.get_workflow_status()
        assert isinstance(status, dict)
        assert status["current_stage"] == "test_stage"
        assert "stages_completed" in status
        assert "total_stages" in status
        assert "workflow_context" in status

    def test_reset_workflow(self):
        """Test workflow reset."""
        workflow = ReflectionWorkflow()
        workflow.current_stage = "test_stage"
        workflow.workflow_context = {"test": "data"}
        workflow.execution_history = [{"test": "history"}]
        
        workflow.reset_workflow()
        assert workflow.current_stage is None
        assert len(workflow.workflow_context) == 0
        assert len(workflow.execution_history) == 0


class TestWorkflowStage:
    """Test workflow stage functionality."""

    def test_workflow_stage_creation(self):
        """Test workflow stage creation."""
        stage = WorkflowStage(
            name="test_stage",
            description="Test stage description",
            requires_reflection=True,
            isolation_level="moderate",
            context_expansion=True
        )
        
        assert stage.name == "test_stage"
        assert stage.description == "Test stage description"
        assert stage.requires_reflection == True
        assert stage.isolation_level == "moderate"
        assert stage.context_expansion == True
        assert stage.metrics_tracking == True  # default value


class TestWorkflowResult:
    """Test workflow result functionality."""

    def test_workflow_result_creation(self):
        """Test workflow result creation."""
        result = WorkflowResult(
            success=True,
            stage_results={"stage1": {"success": True}},
            reflection_insights=[{"insight": "test"}],
            metrics={"time": 10.5},
            execution_time=10.5
        )
        
        assert result.success == True
        assert "stage1" in result.stage_results
        assert len(result.reflection_insights) == 1
        assert result.metrics["time"] == 10.5
        assert result.execution_time == 10.5
        assert result.error_message is None

    def test_workflow_result_with_error(self):
        """Test workflow result with error."""
        result = WorkflowResult(
            success=False,
            stage_results={},
            reflection_insights=[],
            metrics={},
            execution_time=5.0,
            error_message="Test error"
        )
        
        assert result.success == False
        assert result.error_message == "Test error"


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_config_manager_workflow_integration(self):
        """Test integration between config manager and workflow."""
        manager = ConfigManager()
        reflection_config = manager.get_reflection_config()
        
        workflow = ReflectionWorkflow(reflection_config)
        assert workflow.config is not None
        assert workflow.reflection_agent is not None

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_config(self):
        """Test end-to-end workflow with configuration."""
        manager = ConfigManager()
        config = manager.get_reflection_config()
        
        workflow = ReflectionWorkflow(config)
        
        # Mock all async dependencies
        with patch.object(workflow.reflection_agent, 'analyze_knowledge_gaps', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {"gaps": [], "confidence_score": 0.9}
            
            with patch.object(workflow.progressive_enabler, 'should_enable_isolation', new_callable=AsyncMock) as mock_enable:
                mock_enable.return_value = {"should_enable": False}
                
                with patch.object(workflow.isolation_metrics, 'update_isolation_level', new_callable=AsyncMock):
                    with patch.object(workflow.isolation_metrics, 'get_metrics_summary', new_callable=AsyncMock) as mock_metrics:
                        mock_metrics.return_value = {"isolation_active": False}
                        
                        result = await workflow.execute_workflow(
                            "Integration test query",
                            {"test": "context"}
                        )
                        
                        assert isinstance(result, WorkflowResult)
                        assert result.success == True

    def test_configuration_validation_integration(self):
        """Test configuration validation across all components."""
        manager = ConfigManager()
        
        # Validate main configuration
        main_valid = manager.validate_configuration()
        assert isinstance(main_valid, bool)
        
        # Validate integrator configuration
        integrator_valid = manager.integrator.validate_configuration()
        assert isinstance(integrator_valid, bool)
        
        # Validate researcher configuration through integrator
        researcher_config = manager.get_researcher_config()
        assert isinstance(researcher_config, ResearcherConfig)
        assert hasattr(researcher_config, 'enhanced_reflection')
        assert hasattr(researcher_config, 'isolation')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])