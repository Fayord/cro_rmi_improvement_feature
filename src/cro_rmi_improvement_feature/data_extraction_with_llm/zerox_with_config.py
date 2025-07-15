"""
Example of how to integrate the configuration wrapper with zerox
to handle multiple LLM configurations without environment variable conflicts.
"""

import os
import asyncio
from dotenv import load_dotenv
from config_wrapper import LLMConfigManager, LLMConfig, LLMProvider

# Load environment variables
dir_path = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(dir_path, "../../../.env")
load_dotenv(env_path)


def create_llm_configs():
    """Create LLM configurations for your chatbot"""
    config_manager = LLMConfigManager()

    # Add your Azure OpenAI configurations
    config_manager.add_config(
        LLMConfig(
            name="azure-4o-mini",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4o-mini",
            api_key=os.getenv("AZURE_API_KEY_4O_MINI"),  # Use different env var names
            api_base=os.getenv("AZURE_API_BASE_4O_MINI"),
            api_version="2023-05-15",
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="azure-4.1-mini",
            provider=LLMProvider.AZURE,
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("AZURE_API_KEY_4_1_MINI"),  # Use different env var names
            api_base=os.getenv("AZURE_API_BASE_4_1_MINI"),
            api_version="2023-05-15",
        )
    )

    # Add other providers if needed
    config_manager.add_config(
        LLMConfig(
            name="openai-gpt4",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    return config_manager


async def process_document_with_config(
    file_path: str, config_name: str, output_dir: str
):
    """
    Process a document using zerox with a specific LLM configuration

    Args:
        file_path: Path to the document to process
        config_name: Name of the LLM configuration to use
        output_dir: Directory to save results
    """
    from pyzerox import zerox

    config_manager = create_llm_configs()

    # Get the configuration
    config = config_manager.get_config(config_name)
    if not config:
        raise ValueError(f"Configuration '{config_name}' not found")

    # Use the configuration with zerox
    with config_manager.use_config(config_name):
        # Now zerox will use the environment variables set by the config
        result = await zerox(
            file_path=file_path,
            model=config.model,
            output_dir=output_dir,
            custom_system_prompt=None,
            select_pages=None,
        )

    return result


async def process_multiple_documents():
    """Example of processing multiple documents with different configurations"""

    # Example file paths
    files_to_process = [
        f"{dir_path}/2024-TRUE-annual_report.pdf",
        f"{dir_path}/วิธีการสมัครเป็น IB PC.V.pdf",
    ]

    # Different configurations for different files
    configs_to_use = ["azure-4o-mini", "azure-4.1-mini"]

    results = []

    for i, file_path in enumerate(files_to_process):
        config_name = configs_to_use[i % len(configs_to_use)]
        output_dir = f"{dir_path}/output_test/{config_name}"

        print(f"Processing {file_path} with {config_name}")

        try:
            result = await process_document_with_config(
                file_path, config_name, output_dir
            )
            results.append({"file": file_path, "config": config_name, "result": result})
            print(f"✅ Successfully processed with {config_name}")
        except Exception as e:
            print(f"❌ Error processing with {config_name}: {e}")

    return results


def demonstrate_config_switching():
    """Demonstrate how environment variables are switched"""
    config_manager = create_llm_configs()

    print("Available configurations:", config_manager.list_configs())
    print("\n" + "=" * 50)

    # Show environment before switching
    print("Environment before switching:")
    print(f"AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
    print(f"AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")

    # Test switching to first config
    with config_manager.use_config("azure-4o-mini"):
        print("\nEnvironment with azure-4o-mini:")
        print(f"AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
        print(f"AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")

    # Test switching to second config
    with config_manager.use_config("azure-4.1-mini"):
        print("\nEnvironment with azure-4.1-mini:")
        print(f"AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
        print(f"AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")

    # Show environment after switching (should be restored)
    print("\nEnvironment after switching (restored):")
    print(f"AZURE_API_KEY: {os.environ.get('AZURE_API_KEY', 'Not set')}")
    print(f"AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'Not set')}")


if __name__ == "__main__":
    # Demonstrate the configuration switching
    print("Demonstrating configuration wrapper with zerox...")
    demonstrate_config_switching()

    # Uncomment to actually process documents
    # results = asyncio.run(process_multiple_documents())
    # print(f"Processed {len(results)} documents")
