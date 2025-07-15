import os
import pytest
from config_wrapper import LLMConfigManager, LLMConfig, LLMProvider


def test_config_manager_basic_operations():
    """Test basic operations of the configuration manager"""
    config_manager = LLMConfigManager()

    # Test adding configurations
    config1 = LLMConfig(
        name="test-azure-4o",
        provider=LLMProvider.AZURE,
        model="azure/gpt-4o-mini",
        api_key="test-key-4o",
        api_base="https://test-4o.openai.azure.com",
        api_version="2023-05-15",
    )

    config2 = LLMConfig(
        name="test-azure-4.1",
        provider=LLMProvider.AZURE,
        model="azure/gpt-4.1-mini",
        api_key="test-key-4.1",
        api_base="https://test-4.1.openai.azure.com",
        api_version="2023-05-15",
    )

    config_manager.add_config(config1)
    config_manager.add_config(config2)

    # Test listing configurations
    configs = config_manager.list_configs()
    assert "test-azure-4o" in configs
    assert "test-azure-4.1" in configs
    assert len(configs) == 2

    # Test getting configurations
    retrieved_config = config_manager.get_config("test-azure-4o")
    assert retrieved_config is not None
    assert retrieved_config.api_key == "test-key-4o"
    assert retrieved_config.model == "azure/gpt-4o-mini"


def test_environment_variable_switching():
    """Test that environment variables are properly switched and restored"""
    config_manager = LLMConfigManager()

    # Add test configurations
    config_manager.add_config(
        LLMConfig(
            name="test-azure-4o",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4o-mini",
            api_key="test-key-4o",
            api_base="https://test-4o.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="test-azure-4.1",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4.1-mini",
            api_key="test-key-4.1",
            api_base="https://test-4.1.openai.azure.com",
            api_version="2023-05-15",
        )
    )

    # Store original environment variables
    original_azure_key = os.environ.get("AZURE_API_KEY")
    original_azure_base = os.environ.get("AZURE_API_BASE")

    try:
        # Test first configuration
        with config_manager.use_config("test-azure-4o"):
            assert os.environ.get("AZURE_API_KEY") == "test-key-4o"
            assert (
                os.environ.get("AZURE_API_BASE") == "https://test-4o.openai.azure.com"
            )
            assert os.environ.get("AZURE_API_VERSION") == "2023-05-15"

        # Verify environment is restored
        assert os.environ.get("AZURE_API_KEY") == original_azure_key
        assert os.environ.get("AZURE_API_BASE") == original_azure_base

        # Test second configuration
        with config_manager.use_config("test-azure-4.1"):
            assert os.environ.get("AZURE_API_KEY") == "test-key-4.1"
            assert (
                os.environ.get("AZURE_API_BASE") == "https://test-4.1.openai.azure.com"
            )
            assert os.environ.get("AZURE_API_VERSION") == "2023-05-15"

        # Verify environment is restored again
        assert os.environ.get("AZURE_API_KEY") == original_azure_key
        assert os.environ.get("AZURE_API_BASE") == original_azure_base

    finally:
        # Clean up any test environment variables
        if original_azure_key is None and "AZURE_API_KEY" in os.environ:
            del os.environ["AZURE_API_KEY"]
        if original_azure_base is None and "AZURE_API_BASE" in os.environ:
            del os.environ["AZURE_API_BASE"]


def test_execute_with_config():
    """Test the execute_with_config method"""
    config_manager = LLMConfigManager()

    config_manager.add_config(
        LLMConfig(
            name="test-openai",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-openai-key",
        )
    )

    def test_function():
        return os.environ.get("OPENAI_API_KEY")

    # Store original environment
    original_openai_key = os.environ.get("OPENAI_API_KEY")

    try:
        # Test execution with config
        result = config_manager.execute_with_config("test-openai", test_function)
        assert result == "test-openai-key"

        # Verify environment is restored
        assert os.environ.get("OPENAI_API_KEY") == original_openai_key

    finally:
        # Clean up
        if original_openai_key is None and "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]


def test_error_handling():
    """Test error handling for invalid configurations"""
    config_manager = LLMConfigManager()

    # Test getting non-existent config
    assert config_manager.get_config("non-existent") is None

    # Test using non-existent config
    with pytest.raises(ValueError, match="Configuration 'non-existent' not found"):
        with config_manager.use_config("non-existent"):
            pass


def test_multiple_providers():
    """Test configurations with different providers"""
    config_manager = LLMConfigManager()

    # Add different provider configurations
    config_manager.add_config(
        LLMConfig(
            name="test-openai",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-openai-key",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="test-gemini",
            provider=LLMProvider.GEMINI,
            model="gemini/gemini-1.5-pro",
            api_key="test-gemini-key",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="test-anthropic",
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-opus-20240229",
            api_key="test-anthropic-key",
        )
    )

    # Store original environment
    original_env = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    }

    try:
        # Test OpenAI
        with config_manager.use_config("test-openai"):
            assert os.environ.get("OPENAI_API_KEY") == "test-openai-key"

        # Test Gemini
        with config_manager.use_config("test-gemini"):
            assert os.environ.get("GEMINI_API_KEY") == "test-gemini-key"

        # Test Anthropic
        with config_manager.use_config("test-anthropic"):
            assert os.environ.get("ANTHROPIC_API_KEY") == "test-anthropic-key"

        # Verify all environments are restored
        for key, value in original_env.items():
            if value is None:
                assert key not in os.environ or os.environ[key] != value
            else:
                assert os.environ.get(key) == value

    finally:
        # Clean up
        for key, original_value in original_env.items():
            if original_value is None and key in os.environ:
                del os.environ[key]


if __name__ == "__main__":
    # Run the tests
    print("Running configuration wrapper tests...")

    test_config_manager_basic_operations()
    print("✅ Basic operations test passed")

    test_environment_variable_switching()
    print("✅ Environment variable switching test passed")

    test_execute_with_config()
    print("✅ Execute with config test passed")

    test_error_handling()
    print("✅ Error handling test passed")

    test_multiple_providers()
    print("✅ Multiple providers test passed")

    print("\n🎉 All tests passed!")
