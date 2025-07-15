import os
import json
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex_ai"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM"""

    name: str
    provider: LLMProvider
    model: str
    api_key: str
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    additional_kwargs: Optional[Dict[str, Any]] = None


class LLMConfigManager:
    """Manages multiple LLM configurations and handles environment variable switching"""

    def __init__(self):
        self.configs: Dict[str, LLMConfig] = {}
        self._original_env_vars = {}

    def add_config(self, config: LLMConfig) -> None:
        """Add a new LLM configuration"""
        self.configs[config.name] = config

    def get_config(self, name: str) -> Optional[LLMConfig]:
        """Get a configuration by name"""
        return self.configs.get(name)

    def list_configs(self) -> list[str]:
        """List all available configuration names"""
        return list(self.configs.keys())

    @contextmanager
    def use_config(self, config_name: str):
        """
        Context manager to temporarily set environment variables for a specific config

        Usage:
            with config_manager.use_config("azure-4o-mini"):
                # Environment variables are set for this config
                result = some_function_that_uses_env_vars()
        """
        config = self.get_config(config_name)
        if not config:
            raise ValueError(f"Configuration '{config_name}' not found")

        # Store original environment variables
        self._backup_env_vars()

        try:
            # Set environment variables based on provider
            self._set_env_vars_for_config(config)
            yield config
        finally:
            # Restore original environment variables
            self._restore_env_vars()

    def _backup_env_vars(self):
        """Backup current environment variables that might be overwritten"""
        env_vars_to_backup = [
            "OPENAI_API_KEY",
            "AZURE_API_KEY",
            "AZURE_API_BASE",
            "AZURE_API_VERSION",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]

        self._original_env_vars = {}
        for var in env_vars_to_backup:
            if var in os.environ:
                self._original_env_vars[var] = os.environ[var]

    def _restore_env_vars(self):
        """Restore original environment variables"""
        for var, value in self._original_env_vars.items():
            os.environ[var] = value

        # Remove variables that weren't originally set
        for var in [
            "OPENAI_API_KEY",
            "AZURE_API_KEY",
            "AZURE_API_BASE",
            "AZURE_API_VERSION",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]:
            if var not in self._original_env_vars and var in os.environ:
                del os.environ[var]

    def _set_env_vars_for_config(self, config: LLMConfig):
        """Set environment variables for a specific configuration"""
        if config.provider == LLMProvider.OPENAI:
            os.environ["OPENAI_API_KEY"] = config.api_key

        elif config.provider == LLMProvider.AZURE:
            os.environ["AZURE_API_KEY"] = config.api_key
            if config.api_base:
                os.environ["AZURE_API_BASE"] = config.api_base
            if config.api_version:
                os.environ["AZURE_API_VERSION"] = config.api_version

        elif config.provider == LLMProvider.GEMINI:
            os.environ["GEMINI_API_KEY"] = config.api_key

        elif config.provider == LLMProvider.ANTHROPIC:
            os.environ["ANTHROPIC_API_KEY"] = config.api_key

        elif config.provider == LLMProvider.VERTEX_AI:
            # For Vertex AI, we might need to set GOOGLE_APPLICATION_CREDENTIALS
            if (
                config.additional_kwargs
                and "vertex_credentials" in config.additional_kwargs
            ):
                # Handle vertex credentials if needed
                pass

    def execute_with_config(self, config_name: str, func, *args, **kwargs):
        """
        Execute a function with a specific configuration

        Usage:
            result = config_manager.execute_with_config("azure-4o-mini", my_function, arg1, arg2)
        """
        with self.use_config(config_name) as config:
            return func(*args, **kwargs)


# Example usage and demonstration
def create_sample_configs():
    """Create sample configurations for demonstration"""
    config_manager = LLMConfigManager()

    # Add Azure OpenAI configurations
    config_manager.add_config(
        LLMConfig(
            name="azure-4o-mini",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4o-mini",
            api_key="your-azure-api-key-4o-mini",
            api_base="https://your-endpoint-4o-mini.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="azure-4.1-mini",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4.1-mini",
            api_key="your-azure-api-key-4.1-mini",
            api_base="https://your-endpoint-4.1-mini.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    # Add OpenAI configuration
    config_manager.add_config(
        LLMConfig(
            name="openai-gpt4",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="your-openai-api-key",
        )
    )

    # Add Gemini configuration
    config_manager.add_config(
        LLMConfig(
            name="gemini-pro",
            provider=LLMProvider.GEMINI,
            model="gemini/gemini-1.5-pro",
            api_key="your-gemini-api-key",
        )
    )

    return config_manager


def sample_function_that_uses_env_vars():
    """Sample function that uses environment variables (like zerox would)"""
    print(f"Current OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'Not set')}")
    print(f"Current AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
    print(f"Current AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")
    print(f"Current GEMINI_API_KEY: {os.environ.get('GEMINI_API_KEY', 'Not set')}")
    return "Function executed successfully"


if __name__ == "__main__":
    # Create configuration manager with sample configs
    config_manager = create_sample_configs()

    print("Available configurations:", config_manager.list_configs())
    print("\n" + "=" * 50)

    # Test different configurations
    for config_name in ["azure-4o-mini", "azure-4.1-mini", "openai-gpt4", "gemini-pro"]:
        print(f"\nTesting configuration: {config_name}")
        print("-" * 30)

        with config_manager.use_config(config_name):
            sample_function_that_uses_env_vars()

        print("Environment restored after context manager")
        print(
            f"AZURE_API_KEY after restore: {os.environ.get('AZURE_API_KEY', 'Not set')}"
        )
