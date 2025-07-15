#!/usr/bin/env python3
"""
Demo script showing how the configuration wrapper works
Run this to see the environment variable switching in action
"""

import os
from config_wrapper import LLMConfigManager, LLMConfig, LLMProvider


def demo_basic_usage():
    """Demonstrate basic usage of the configuration wrapper"""
    print("🚀 Configuration Wrapper Demo")
    print("=" * 50)

    # Create configuration manager
    config_manager = LLMConfigManager()

    # Add different configurations
    config_manager.add_config(
        LLMConfig(
            name="azure-4o-mini",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4o-mini",
            api_key="your-azure-key-4o-mini",
            api_base="https://your-endpoint-4o-mini.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="azure-4.1-mini",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4.1-mini",
            api_key="your-azure-key-4.1-mini",
            api_base="https://your-endpoint-4.1-mini.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="openai-gpt4",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="your-openai-key",
        )
    )

    print(f"Available configurations: {config_manager.list_configs()}")
    print()

    # Show original environment
    print("📋 Original Environment:")
    print(f"  AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
    print(f"  AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")
    print(f"  OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'Not set')}")
    print()

    # Demonstrate switching between configurations
    for config_name in ["azure-4o-mini", "azure-4.1-mini", "openai-gpt4"]:
        print(f"🔄 Switching to: {config_name}")
        print("-" * 30)

        with config_manager.use_config(config_name):
            print(f"  AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
            print(f"  AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")
            print(f"  OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'Not set')}")

            # Simulate a function that uses these environment variables
            print(f"  ✅ Function would use {config_name} configuration")

        print(f"  🔄 Environment restored after {config_name}")
        print()

    # Show final environment (should be same as original)
    print("📋 Final Environment (should match original):")
    print(f"  AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
    print(f"  AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")
    print(f"  OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'Not set')}")
    print()


def demo_execute_with_config():
    """Demonstrate the execute_with_config method"""
    print("🎯 Execute with Config Demo")
    print("=" * 50)

    config_manager = LLMConfigManager()

    # Add a test configuration
    config_manager.add_config(
        LLMConfig(
            name="test-azure",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4o-mini",
            api_key="test-azure-key",
            api_base="https://test-azure.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    def mock_zerox_function():
        """Mock function that uses environment variables like zerox would"""
        return {
            "model": "azure/gpt-4o-mini",
            "api_key": os.environ.get("AZURE_API_KEY"),
            "api_base": os.environ.get("AZURE_API_BASE"),
            "api_version": os.environ.get("AZURE_API_VERSION"),
        }

    # Execute the function with the configuration
    result = config_manager.execute_with_config("test-azure", mock_zerox_function)

    print("Function executed with configuration:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print()


def demo_error_handling():
    """Demonstrate error handling"""
    print("⚠️  Error Handling Demo")
    print("=" * 50)

    config_manager = LLMConfigManager()

    try:
        with config_manager.use_config("non-existent-config"):
            pass
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    print()


if __name__ == "__main__":
    print("Configuration Wrapper Demo")
    print("=" * 60)
    print()

    demo_basic_usage()
    demo_execute_with_config()
    demo_error_handling()

    print("🎉 Demo completed successfully!")
    print("\nKey Benefits:")
    print("✅ No environment variable conflicts")
    print("✅ Easy switching between different LLM configurations")
    print("✅ Automatic cleanup and restoration of environment")
    print("✅ Works with any library that uses environment variables")
    print("✅ Type-safe configuration management")
