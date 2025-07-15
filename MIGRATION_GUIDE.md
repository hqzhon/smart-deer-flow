# Migration Guide: New Architecture Implementation

This guide helps you migrate from the old architecture to the new optimized architecture based on the optimization document.

## Overview

The new architecture implements four key optimizations:

1. **Unified Configuration Management** - Pydantic-based configuration with dependency injection
2. **Modular Graph Components** - Factory pattern for reusable sub-graphs
3. **Standardized Prompt Management** - Centralized prompt management system
4. **Tool Registry** - Automatic tool discovery and registration

## New Components

### 1. Configuration System

**New Files:**
- `src/config/models.py` - Pydantic configuration models
- `src/config/new_config_loader.py` - Enhanced configuration loader

**Usage:**
```python
from src.config.new_config_loader import load_configuration, get_settings

# Load configuration
settings = load_configuration("conf.yaml")

# Access specific configurations
llm_config = settings.get_llm_config()
agent_config = settings.get_agent_config()
```

### 2. Prompt Management

**New Files:**
- `src/prompts/prompt_manager.py` - Centralized prompt management

**Usage:**
```python
from src.prompts.prompt_manager import get_prompt, get_prompt_manager

# Get a prompt
prompt = get_prompt("researcher")

# Get prompt with variables
prompt_with_vars = get_prompt_with_variables("researcher", {"topic": "AI"})
```

### 3. Tool Registry

**New Files:**
- `src/tools/base_tool.py` - Base tool interface and registry
- `src/tools/web_search_tool.py` - New web search tools
- `src/tools/crawl_tool.py` - New crawl tools
- `src/tools/python_repl_tool.py` - New Python REPL tools

**Usage:**
```python
from src.tools.base_tool import get_tool_registry

# Get all available tools
registry = get_tool_registry()
tools = registry.list_tools()

# Get tool schemas for agents
schemas = registry.get_all_schemas()
```

### 4. Modular Graph Components

**New Files:**
- `src/graph/components/reflection_component.py` - Reflection loop component
- `src/graph/components/debate_component.py` - Multi-agent debate component
- `src/graph/components/research_component.py` - Research workflow component

**Usage:**
```python
from src.graph.components.reflection_component import create_reflection_loop
from src.graph.components.research_component import create_research_workflow

# Create modular components
reflection_graph = create_reflection_loop(llm, max_loops=3)
research_graph = create_research_workflow(llm, max_search_results=5)
```

## Migration Steps

### Step 1: Update Configuration

1. **Create new configuration file** based on the example:
   ```bash
   cp conf.yaml.example conf.yaml
   ```

2. **Update configuration format** to use the new structure:
   ```yaml
   # New format
   report_style: "academic"
   llm:
     temperature: 0.7
     timeout: 30
   agents:
     max_search_results: 3
     max_step_num: 5
   reflection:
     enable_enhanced_reflection: true
     max_reflection_loops: 3
   ```

### Step 2: Update Tool Usage

**Old way:**
```python
from src.tools.search import get_web_search_tool
tool = get_web_search_tool(max_results=5)
```

**New way:**
```python
from src.tools.base_tool import get_tool_registry
registry = get_tool_registry()
tool = registry.get_tool("web_search")
```

### Step 3: Update Agent Creation

**Old way:**
```python
from src.agents.agents import create_agent
agent = create_agent(llm, prompt_text)
```

**New way:**
```python
from src.prompts.prompt_manager import get_prompt
from src.agents.agents import create_agent

prompt = get_prompt("researcher")
agent = create_agent(llm, prompt)
```

### Step 4: Update Graph Construction

**Old way:**
```python
from src.graph.builder import create_workflow
workflow = create_workflow()
```

**New way:**
```python
from src.graph.components.reflection_component import create_reflection_loop
from src.graph.components.research_component import create_research_workflow

# Use modular components
reflection_graph = create_reflection_loop(llm)
research_graph = create_research_workflow(llm)
```

## Backward Compatibility

The new architecture maintains backward compatibility:

- **Legacy functions** are still available in `src/tools/__init__.py`
- **Old configuration** can still be used with the legacy system
- **Existing code** will continue to work without changes

## New Entry Point

**New main script:** `main_new.py`

**Usage:**
```bash
# Interactive mode
python main_new.py --interactive

# Single query
python main_new.py "What is quantum computing?"

# With reflection
python main_new.py --enable-reflection "Research AI safety"

# List available tools
python main_new.py --list-tools

# Configuration summary
python main_new.py --config-summary
```

## Configuration Examples

### Basic Configuration
```yaml
report_style: "academic"
agents:
  max_search_results: 3
  max_step_num: 5
  enable_parallel_execution: true

llm:
  temperature: 0.7
  timeout: 30

reflection:
  enable_enhanced_reflection: true
  max_reflection_loops: 3
```

### Advanced Configuration
```yaml
report_style: "technical"
agents:
  max_search_results: 10
  max_step_num: 10
  enable_deep_thinking: true
  enable_parallel_execution: true
  max_parallel_tasks: 5

research:
  enable_researcher_isolation: true
  researcher_isolation_level: "moderate"

advanced_context:
  max_context_ratio: 0.8
  enable_caching: true
  enable_analytics: true
```

## Testing the New Architecture

### 1. Test Configuration Loading
```python
from src.config.new_config_loader import load_configuration
config = load_configuration("conf.yaml")
print(config.get_config_summary())
```

### 2. Test Tool Discovery
```python
from src.tools.base_tool import get_tool_registry
registry = get_tool_registry()
print("Available tools:", registry.list_tools())
```

### 3. Test Prompt Loading
```python
from src.prompts.prompt_manager import get_prompt_manager
manager = get_prompt_manager()
manager.load_prompts()
print("Available prompts:", manager.list_prompts())
```

## Troubleshooting

### Common Issues

1. **Configuration not found**: Ensure `conf.yaml` exists in project root
2. **Tools not discovered**: Check that new tool files are in `src/tools/` directory
3. **Prompts not loaded**: Ensure prompt files exist in `src/prompts/` directory

### Debug Mode

Enable debug logging to see detailed information:
```bash
python main_new.py --debug "your query"
```

## Next Steps

1. **Gradually migrate** existing code to use new components
2. **Test thoroughly** with your specific use cases
3. **Update documentation** for your team
4. **Consider deprecating** old components after full migration

## Support

For issues or questions about the migration:
1. Check the migration guide
2. Enable debug mode for detailed logs
3. Test with simple queries first
4. Verify configuration format matches new schema