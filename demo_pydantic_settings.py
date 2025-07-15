#!/usr/bin/env python3
"""Demo script showing the advantages of pydantic-settings over manual env_mappings."""

import os
from src.config.config_loader import ConfigLoader  # Old system
from src.config.config_loader_v2 import ConfigLoaderV2  # New system
from src.config.settings import AppSettings  # New settings model


def demo_old_vs_new_system():
    """Demonstrate the differences between old and new configuration systems."""
    
    print("=" * 60)
    print("Configuration System Comparison Demo")
    print("=" * 60)
    
    # Set some environment variables for testing
    test_env_vars = {
        'DEER_REPORT_STYLE': 'business',
        'DEER_LLM_TEMPERATURE': '0.8',
        'DEER_LLM_MAX_TOKENS': '2000',
        'DEER_LLM_BASIC_MODEL': 'gpt-4o-mini',  # Required field
        'DEER_MAX_PLAN_ITERATIONS': '5',
        'DEER_ENABLE_DEEP_THINKING': 'true',
        'DEER_ENABLE_RESEARCHER_ISOLATION': 'false',
        'DEER_MAX_REFLECTION_LOOPS': '3',
        'DEER_MCP_ENABLED': 'true',
        'DEER_MCP_TIMEOUT': '45'
    }
    
    # Apply test environment variables
    original_env = {}
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        print("\n1. OLD SYSTEM (Manual env_mappings):")
        print("-" * 40)
        
        # Load with old system
        old_loader = ConfigLoader()
        old_config = old_loader.load_configuration()
        old_dict = old_config.model_dump()
        
        print(f"Report Style: {old_dict.get('report_style')}")
        print(f"LLM Temperature: {old_dict.get('llm', {}).get('temperature')}")
        print(f"LLM Max Tokens: {old_dict.get('llm', {}).get('max_tokens')}")
        print(f"Max Plan Iterations: {old_dict.get('agents', {}).get('max_plan_iterations')}")
        print(f"Enable Deep Thinking: {old_dict.get('agents', {}).get('enable_deep_thinking')}")
        print(f"Researcher Isolation: {old_dict.get('research', {}).get('enable_researcher_isolation')}")
        print(f"Max Reflection Loops: {old_dict.get('reflection', {}).get('max_reflection_loops')}")
        print(f"MCP Enabled: {old_dict.get('mcp', {}).get('enabled')}")
        print(f"MCP Timeout: {old_dict.get('mcp', {}).get('timeout')}")
        
        print("\n2. NEW SYSTEM (pydantic-settings):")
        print("-" * 40)
        
        # Load with new system
        new_loader = ConfigLoaderV2()
        new_config = new_loader.load_configuration()
        
        print(f"Report Style: {new_config.report_style}")
        print(f"LLM Temperature: {new_config.llm.temperature}")
        print(f"LLM Max Tokens: {new_config.llm.max_tokens}")
        print(f"Max Plan Iterations: {new_config.agents.max_plan_iterations}")
        print(f"Enable Deep Thinking: {new_config.agents.enable_deep_thinking}")
        print(f"Researcher Isolation: {new_config.research.enable_researcher_isolation}")
        print(f"Max Reflection Loops: {new_config.reflection.max_reflection_loops}")
        print(f"MCP Enabled: {new_config.mcp.enabled}")
        print(f"MCP Timeout: {new_config.mcp.timeout}")
        
        print("\n3. ADVANTAGES OF NEW SYSTEM:")
        print("-" * 40)
        print("✅ No manual env_mappings dictionary to maintain")
        print("✅ Automatic environment variable discovery")
        print("✅ Type conversion handled automatically")
        print("✅ Support for nested environment variables (DEER_LLM__TEMPERATURE)")
        print("✅ Case-insensitive environment variables")
        print("✅ Better validation and error messages")
        print("✅ Future-proof: new fields automatically work with env vars")
        print("✅ Cleaner, more maintainable code")
        
        print("\n4. TESTING NEW FIELD ADDITION:")
        print("-" * 40)
        
        # Demonstrate adding a new field that works automatically
        os.environ['DEER_NEW_FEATURE_ENABLED'] = 'true'
        
        # This would work automatically with pydantic-settings if we added
        # new_feature_enabled: bool = False to any settings model
        print("If we add 'new_feature_enabled: bool = False' to a settings model,")
        print("DEER_NEW_FEATURE_ENABLED=true would automatically work!")
        print("No need to update any env_mappings dictionary.")
        
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        
        # Clean up test env var
        os.environ.pop('DEER_NEW_FEATURE_ENABLED', None)


def demo_direct_settings_usage():
    """Demonstrate direct usage of the new settings classes."""
    
    print("\n" + "=" * 60)
    print("Direct Settings Usage Demo")
    print("=" * 60)
    
    # Set some environment variables
    os.environ['DEER_REPORT_STYLE'] = 'technical'
    os.environ['DEER_LLM_TEMPERATURE'] = '0.3'
    os.environ['DEER_LLM_BASIC_MODEL'] = 'gpt-4o-mini'
    
    try:
        # Direct instantiation - automatically loads from environment
        settings = AppSettings()
        
        print(f"Report Style: {settings.report_style}")
        print(f"LLM Basic Model: {settings.llm.basic_model}")
        print(f"LLM Temperature: {settings.llm.temperature}")
        print(f"Database Connection: {settings.database.connection_string or 'Not configured'}")
        
        print("\n✨ All environment variables loaded automatically!")
        print("✨ No manual mapping required!")
        
    finally:
        # Clean up
        for key in ['DEER_REPORT_STYLE', 'DEER_LLM_TEMPERATURE', 'DEER_LLM_BASIC_MODEL']:
            os.environ.pop(key, None)


if __name__ == "__main__":
    demo_old_vs_new_system()
    demo_direct_settings_usage()
    
    print("\n" + "=" * 60)
    print("Demo completed! 🎉")
    print("The new pydantic-settings system is ready for use.")
    print("=" * 60)