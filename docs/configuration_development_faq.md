# Configuration Development FAQ

## Overview

This FAQ provides practical guidance for developers working with DeerFlow's unified configuration system. Find answers to common questions and best practices for configuration development.

## Quick Start

### Q: How do I access configuration in my code?

**A:** Use the configuration service for type-safe access:

```python
from src.config.config_service import get_config_service

# Get the configuration service
config = get_config_service()

# Access specific settings
search_engine = config.get_search_engine()
llm_type = config.get_agent_llm_type('researcher')
perf_settings = config.get_performance_settings()
```

### Q: What's the difference between config service and direct settings access?

**A:** Both approaches work, choose based on your needs:

```python
# Method 1: Configuration Service (Recommended)
from src.config.config_service import get_config_service
config = get_config_service()
search_engine = config.get_search_engine()  # Returns enum value

# Method 2: Direct Settings Access
from src.config.config_service import get_app_settings
app_settings = get_app_settings()
search_engine = app_settings.tools.search_engine  # Returns raw value
```

**Use config service when:** You need validation, enum conversion, or helper methods  
**Use direct access when:** You need raw values or performance is critical

## Environment Variables

### Q: What environment variables are supported?

**A:** DeerFlow supports both legacy and new environment variable formats:

```bash
# Legacy format (still supported)
SELECTED_SEARCH_ENGINE=tavily
SELECTED_RAG_PROVIDER=ragflow

# New DEER_ prefixed format (recommended)
DEER_TOOLS_SEARCH_ENGINE=tavily
DEER_TOOLS_RAG_PROVIDER=ragflow
DEER_ENABLE_ADVANCED_OPTIMIZATION=true
DEER_MAX_CONNECTIONS=50
DEER_BATCH_SIZE=10

# Agent LLM mappings
DEER_COORDINATOR_LLM=basic
DEER_RESEARCHER_LLM=reasoning
DEER_PLANNER_LLM=advanced
```

### Q: How do I set environment variables for different environments?

**A:** Use `.env` files or system environment variables:

```bash
# Development (.env.development)
DEER_TOOLS_SEARCH_ENGINE=duckduckgo
DEER_ENABLE_ADVANCED_OPTIMIZATION=false
DEER_LOG_LEVEL=debug

# Production (.env.production)
DEER_TOOLS_SEARCH_ENGINE=tavily
DEER_ENABLE_ADVANCED_OPTIMIZATION=true
DEER_LOG_LEVEL=info
```

### Q: What happens if I set both legacy and new environment variables?

**A:** New DEER_ prefixed variables take precedence over legacy ones.

## Configuration Files

### Q: Where should I put my configuration files?

**A:** Follow this structure:

```
config/
├── app_config.yaml              # Main configuration
├── development_config.yaml      # Development overrides
├── production_config.yaml       # Production overrides
└── test_config.yaml             # Test environment
```

### Q: How do I create a custom configuration file?

**A:** Create a YAML file following the schema:

```yaml
# custom_config.yaml
tools:
  search_engine: "tavily"
  rag_provider: "ragflow"

performance:
  enable_advanced_optimization: true
  connection_pool:
    max_connections: 100
  batch_processing:
    batch_size: 20

agent_llm_map:
  coordinator: "basic"
  researcher: "reasoning"
  planner: "advanced"
```

### Q: How do I load a custom configuration file?

**A:** Use the config service reload method:

```python
from src.config.config_service import get_config_service

config = get_config_service()
config.reload_config('custom_config.yaml')
```

## Development Patterns

### Q: How do I add a new configuration option?

**A:** Follow these steps:

1. **Add to the model** in `src/config/models.py`:
```python
class ToolsSettings(BaseSettings):
    search_engine: SearchEngine = SearchEngine.TAVILY
    rag_provider: str = "ragflow"
    new_feature_enabled: bool = False  # Add this
```

2. **Add environment variable support**:
```python
class ToolsSettings(BaseSettings):
    # ... existing fields ...
    new_feature_enabled: bool = Field(
        default=False,
        env="DEER_NEW_FEATURE_ENABLED"
    )
```

3. **Add to config service** if needed:
```python
def is_new_feature_enabled(self) -> bool:
    return self.app_settings.tools.new_feature_enabled
```

### Q: How do I validate configuration values?

**A:** Use Pydantic validators:

```python
from pydantic import validator

class PerformanceSettings(BaseSettings):
    max_connections: int = 50
    
    @validator('max_connections')
    def validate_max_connections(cls, v):
        if v <= 0:
            raise ValueError('max_connections must be positive')
        if v > 1000:
            raise ValueError('max_connections cannot exceed 1000')
        return v
```

### Q: How do I handle configuration in tests?

**A:** Use test-specific configuration:

```python
# In your test
from src.config.config_service import get_config_service

def test_with_custom_config():
    config = get_config_service()
    
    # Option 1: Reload with test config
    config.reload_config('test_config.yaml')
    
    # Option 2: Override specific values
    original_search_engine = config.get_search_engine()
    config.app_settings.tools.search_engine = SearchEngine.DUCKDUCKGO
    
    # Your test code here
    
    # Restore original value
    config.app_settings.tools.search_engine = original_search_engine
```

## Troubleshooting

### Q: I'm getting "Configuration validation failed" errors. How do I debug?

**A:** Enable debug logging and check validation:

```python
from src.config.config_service import get_config_service
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

config = get_config_service()

# Check validation
if not config.validate_config():
    print("Configuration validation failed")
    # Check individual settings
    try:
        app_settings = get_app_settings()
        print(f"Current settings: {app_settings}")
    except Exception as e:
        print(f"Error loading settings: {e}")
```

### Q: My environment variables aren't being picked up. What's wrong?

**A:** Check these common issues:

1. **Variable naming**: Ensure you're using the correct format
2. **Data types**: Boolean values should be `true`/`false`, not `True`/`False`
3. **Loading order**: Environment variables override config files
4. **Restart required**: Some changes require application restart

```bash
# Correct
DEER_ENABLE_ADVANCED_OPTIMIZATION=true

# Incorrect
DEER_ENABLE_ADVANCED_OPTIMIZATION=True
```

### Q: How do I see what configuration values are currently loaded?

**A:** Use the config service debug methods:

```python
from src.config.config_service import get_config_service

config = get_config_service()

# Print current configuration
print(f"Search Engine: {config.get_search_engine()}")
print(f"RAG Provider: {config.app_settings.tools.rag_provider}")
print(f"Performance Settings: {config.get_performance_settings()}")

# Check if features are enabled
print(f"Advanced Optimization: {config.is_feature_enabled('performance.enable_advanced_optimization')}")
```

### Q: I'm getting import errors after updating configuration code. What should I do?

**A:** Update your imports to use the new configuration system:

```python
# Old imports (remove these)
# from src.config.tools import SELECTED_SEARCH_ENGINE
# from src.config.agents import AGENT_LLM_MAP
# from src.config.performance_config import PerformanceConfig

# New imports (use these)
from src.config.config_service import get_config_service, get_app_settings
from src.config.models import SearchEngine, AgentLLMType
```

## Performance Considerations

### Q: Is the configuration service expensive to call repeatedly?

**A:** The config service is designed to be lightweight:

- Configuration is loaded once and cached
- Subsequent calls return cached values
- Hot-reloading only occurs when explicitly requested

```python
# This is fine - config is cached
for i in range(1000):
    config = get_config_service()
    search_engine = config.get_search_engine()
```

### Q: When should I reload configuration?

**A:** Only reload when necessary:

- **Development**: When testing different configurations
- **Production**: When configuration files are updated externally
- **Never**: In tight loops or frequently called functions

```python
# Good: Reload when needed
if config_file_changed:
    config.reload_config()

# Bad: Reload in loops
for item in items:
    config.reload_config()  # Don't do this!
    process(item)
```

## Best Practices

### Q: What are the configuration best practices?

**A:** Follow these guidelines:

1. **Use environment variables for secrets and environment-specific values**
2. **Use config files for application structure and defaults**
3. **Validate configuration early in application startup**
4. **Use enums for predefined choices**
5. **Document configuration options with docstrings**
6. **Test configuration changes thoroughly**

```python
# Good: Environment-specific
DEER_DATABASE_URL=postgresql://localhost/dev_db
DEER_API_KEY=secret_key

# Good: Application defaults in config file
tools:
  search_engine: "tavily"
  timeout: 30
```

### Q: How do I handle sensitive configuration data?

**A:** Use environment variables and never commit secrets:

```python
# In your model
class DatabaseSettings(BaseSettings):
    url: str = Field(..., env="DEER_DATABASE_URL")
    password: str = Field(..., env="DEER_DB_PASSWORD")
    
    class Config:
        # Don't log sensitive fields
        fields = {
            'password': {'write_only': True}
        }
```

```bash
# In .env (never commit this file)
DEER_DATABASE_URL=postgresql://user:pass@localhost/db
DEER_DB_PASSWORD=secret_password
```

This FAQ covers the most common configuration development scenarios. For additional help, check the configuration models in `src/config/models.py` and the service implementation in `src/config/config_service.py`.