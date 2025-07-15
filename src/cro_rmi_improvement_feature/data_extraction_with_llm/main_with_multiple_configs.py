"""
Main file implementing Option B: Multiple Configurations
Process multiple documents with different LLM configurations
"""

from pyzerox import zerox
import os
import json
import asyncio
from dotenv import load_dotenv
import pickle
from config_wrapper import LLMConfigManager, LLMConfig, LLMProvider

dir_path = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(dir_path, "../../../.env")
load_dotenv(env_path)


def create_zerox_configs():
    """Create LLM configurations for zerox processing"""
    config_manager = LLMConfigManager()

    # Add OpenAI configurations
    config_manager.add_config(
        LLMConfig(
            name="zerox-4o-mini",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    config_manager.add_config(
        LLMConfig(
            name="zerox-4.1-nano",
            provider=LLMProvider.OPENAI,
            model="gpt-4.1-nano",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    # Add failing configuration (invalid model)
    config_manager.add_config(
        LLMConfig(
            name="zerox-5o-mini",
            provider=LLMProvider.OPENAI,
            model="gpt-5o-mini",  # This model doesn't exist, will fail
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    return config_manager


async def process_single_document(file_path: str, config_name: str, output_dir: str):
    """
    Process a single document with a specific configuration

    Args:
        file_path: Path to the document
        config_name: Name of the configuration to use
        output_dir: Directory to save results

    Returns:
        Dictionary with processing results
    """
    config_manager = create_zerox_configs()

    # Get the configuration
    config = config_manager.get_config(config_name)
    if not config:
        raise ValueError(f"Configuration '{config_name}' not found")

    print(f"🔄 Processing {os.path.basename(file_path)} with {config_name}")
    print(f"   Model: {config.model}")
    print(f"   Provider: {config.provider.value}")

    try:
        # Use the configuration wrapper
        with config_manager.use_config(config_name):
            result = await zerox(
                file_path=file_path,
                model=config.model,
                output_dir=output_dir,
                custom_system_prompt=None,
                select_pages=None,
            )

        # Save result to pickle
        file_name = os.path.basename(file_path)
        pickle_path = f"{output_dir}/{file_name}.pkl"
        save_result_to_pickle(result, pickle_path)

        print(f"✅ Successfully processed with {config_name}")
        return {
            "file": file_path,
            "config": config_name,
            "model": config.model,
            "result": result,
            "pickle_path": pickle_path,
            "status": "success",
        }

    except Exception as e:
        print(f"❌ Error processing with {config_name}: {e}")
        return {
            "file": file_path,
            "config": config_name,
            "model": config.model,
            "error": str(e),
            "status": "error",
        }


async def process_multiple_documents():
    """
    Process multiple documents with different configurations
    This is Option B implementation
    """
    config_manager = create_zerox_configs()

    # Available files to process
    available_files = [
        # f"{dir_path}/2024-TRUE-annual_report.pdf",
        # f"{dir_path}/วิธีการสมัครเป็น IB PC.V.pdf",
        f"{dir_path}/เปิดบัญชีกับ LX Phone V.pdf",
        f"{dir_path}/เปิดบัญชีกับ LX Phone V.pdf",
        f"{dir_path}/เปิดบัญชีกับ LX Phone V.pdf",
        # f"{dir_path}/AUDITOR_REPORT.DOCX",
        # f"{dir_path}/Screenshot.png",
    ]

    # Available configurations
    configs = ["zerox-4o-mini", "zerox-4.1-nano", "zerox-5o-mini"]

    print("🚀 Starting multi-configuration document processing")
    print(f"Available files: {len(available_files)}")
    print(f"Available configs: {configs}")
    print("=" * 60)

    results = []

    for i, file_path in enumerate(available_files):
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        # Choose configuration (round-robin)
        config_name = configs[i % len(configs)]

        # Create output directory for this config
        output_dir = f"{dir_path}/output_test/{config_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Process the document
        result = await process_single_document(file_path, config_name, output_dir)
        results.append(result)

        print("-" * 40)

    return results


def save_result_to_pickle(result, file_path):
    """Save result to pickle file"""
    with open(file_path, "wb") as f:
        pickle.dump(result, f)


def print_processing_summary(results):
    """Print a summary of the processing results"""
    print("\n" + "=" * 60)
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"📄 Total: {len(results)}")

    if successful:
        print("\n✅ Successfully processed:")
        for result in successful:
            print(f"   • {os.path.basename(result['file'])} → {result['config']}")

    if failed:
        print("\n❌ Failed to process:")
        for result in failed:
            print(
                f"   • {os.path.basename(result['file'])} → {result['config']}: {result['error']}"
            )


async def main():
    """Main entry point"""
    print("🎯 Option B: Multiple Configuration Document Processing")
    print("=" * 60)

    # Check if configurations are available
    config_manager = create_zerox_configs()
    available_configs = config_manager.list_configs()

    print(f"Available configurations: {available_configs}")

    # Check environment variables
    print("\n🔍 Environment Check:")
    for config_name in available_configs:
        config = config_manager.get_config(config_name)
        if config:
            print(f"   {config_name}: {config.provider.value} - {config.model}")

    print("\n" + "=" * 60)

    # Process documents
    results = await process_multiple_documents()

    # Print summary
    print_processing_summary(results)

    return results


if __name__ == "__main__":
    # Run the main function
    results = asyncio.run(main())

    print(f"\n🎉 Processing complete! Check the output_test directory for results.")
