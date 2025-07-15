# Configure PDF conversion options BEFORE importing zerox
# This monkey patches the default options
import sys
import os
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and modify the default options
from pyzerox.constants.conversion import PDFConversionDefaultOptions

# =============================================================================
# PDF CONVERSION CONFIGURATION
# =============================================================================


class PDFConversionConfig:
    """Configuration class for PDF conversion settings"""

    # Preset configurations for different use cases
    PRESETS = {
        "fast": {
            "dpi": 150,
            "format": "jpeg",
            "size": (None, 800),
            "thread_count": 4,
            "use_pdftocairo": True,
            "compression": 85,
            "description": "Fast processing, lower quality",
        },
        "balanced": {
            "dpi": 300,
            "format": "png",
            "size": (None, 1056),
            "thread_count": 2,
            "use_pdftocairo": True,
            "compression": None,
            "description": "Balanced quality and speed",
        },
        "high_quality": {
            "dpi": 600,
            "format": "png",
            "size": (None, 2112),
            "thread_count": 2,
            "use_pdftocairo": True,
            "compression": None,
            "description": "High quality for complex documents",
        },
        "ultra_quality": {
            "dpi": 900,
            "format": "png",
            "size": (None, 3000),
            "thread_count": 1,
            "use_pdftocairo": True,
            "compression": None,
            "description": "Ultra high quality for detailed analysis",
        },
        "memory_efficient": {
            "dpi": 200,
            "format": "jpeg",
            "size": (None, 600),
            "thread_count": 1,
            "use_pdftocairo": False,
            "compression": 70,
            "description": "Memory efficient for large documents",
        },
    }

    def __init__(
        self,
        preset: str = "high_quality",
        custom_settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PDF conversion configuration

        Args:
            preset: Name of preset to use ("fast", "balanced", "high_quality", "ultra_quality", "memory_efficient")
            custom_settings: Optional dict to override specific settings
        """
        if preset not in self.PRESETS:
            raise ValueError(
                f"Invalid preset '{preset}'. Available presets: {list(self.PRESETS.keys())}"
            )

        self.settings = self.PRESETS[preset].copy()

        # Apply custom settings if provided
        if custom_settings:
            self.settings.update(custom_settings)

    def apply_to_zerox(self):
        """Apply the configuration to zerox PDF conversion defaults"""
        PDFConversionDefaultOptions.DPI = self.settings["dpi"]
        PDFConversionDefaultOptions.FORMAT = self.settings["format"]
        PDFConversionDefaultOptions.SIZE = self.settings["size"]
        PDFConversionDefaultOptions.THREAD_COUNT = self.settings["thread_count"]
        PDFConversionDefaultOptions.USE_PDFTOCAIRO = self.settings["use_pdftocairo"]

    def print_settings(self):
        """Print current PDF conversion settings"""
        print(f"📄 PDF Conversion Settings:")
        print(f"   Preset: {self.settings.get('description', 'Custom')}")
        print(f"   DPI: {self.settings['dpi']}")
        print(f"   Format: {self.settings['format']}")
        print(f"   Size: {self.settings['size']}")
        print(f"   Thread Count: {self.settings['thread_count']}")
        print(f"   Use PDFToCairo: {self.settings['use_pdftocairo']}")
        if self.settings.get("compression"):
            print(f"   Compression: {self.settings['compression']}%")
        print(f"   Estimated Quality: {self._get_quality_estimate()}")

    def _get_quality_estimate(self) -> str:
        """Estimate quality based on settings"""
        dpi = self.settings["dpi"]
        size = self.settings["size"][1] if self.settings["size"][1] else 1000

        if dpi >= 600 and size >= 2000:
            return "Excellent"
        elif dpi >= 300 and size >= 1000:
            return "Good"
        elif dpi >= 200 and size >= 800:
            return "Fair"
        else:
            return "Basic"

    @classmethod
    def list_presets(cls):
        """List all available presets with descriptions"""
        print("🎯 Available PDF Conversion Presets:")
        for name, config in cls.PRESETS.items():
            print(f"   {name:15} - {config['description']}")
            print(
                f"                DPI: {config['dpi']}, Size: {config['size']}, Format: {config['format']}"
            )


# =============================================================================
# CONFIGURE PDF CONVERSION
# =============================================================================

# Choose your preset or create custom settings
# Available presets: "fast", "balanced", "high_quality", "ultra_quality", "memory_efficient"

# Option 1: Use a preset
preset = "high_quality"
pdf_config = PDFConversionConfig(preset=preset)

# Option 2: Use preset with custom overrides
# pdf_config = PDFConversionConfig(
#     preset="high_quality",
#     custom_settings={
#         "dpi": 450,  # Override DPI
#         "thread_count": 1,  # Override thread count
#     }
# )

# Option 3: Create completely custom settings
# pdf_config = PDFConversionConfig(
#     preset="balanced",  # Start with balanced preset
#     custom_settings={
#         "dpi": 400,
#         "format": "jpeg",
#         "size": (None, 1500),
#         "thread_count": 3,
#         "use_pdftocairo": True,
#         "compression": 90,
#     }
# )

# Apply the configuration
pdf_config.apply_to_zerox()
pdf_config.print_settings()

# List all available presets (uncomment to see options)
# PDFConversionConfig.list_presets()
from pyzerox import zerox
import os
import json
import asyncio
from dotenv import load_dotenv
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pyzerox.core.types import Page


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost based on tokens and model"""
    costs = {
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gemini/gemini-2.5-flash": {"input": 0.3, "output": 2.50},
        "gemini/gemini-2.5-flash-lite-preview-06-17": {
            "input": 0.1,
            "output": 0.4,
        },
    }

    if model not in costs:
        return 0.0  # Unknown model

    cost = (input_tokens * costs[model]["input"] / 1_000_000) + (
        output_tokens * costs[model]["output"] / 1_000_000
    )
    return round(cost, 4)


dir_path = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(dir_path, "../../../.env")
load_dotenv(env_path)

### Model Setup (Use only Vision Models) Refer: https://docs.litellm.ai/docs/providers ###

## placeholder for additional model kwargs which might be required for some models
kwargs = {}

## system prompt to use for the vision model
custom_system_prompt = None

# to override
# custom_system_prompt = "For the below PDF page, do something..something..." ## example

# Define list of models to process
model_list = [
    "gpt-4o-mini",  ## openai model
    "gpt-4.1-nano",  ## openai model
    "gpt-4.1-mini",
    "gemini/gemini-2.5-flash-lite-preview-06-17",
    "gemini/gemini-2.5-flash",
]


@dataclass
class ZeroxOutput:
    """
    Dataclass to store the output of the Zerox class.
    """

    completion_time: float
    file_name: str
    input_tokens: int
    output_tokens: int
    pages: List[Page]


def get_output_file_path(file_path: str, model: str, preset: str) -> str:
    """Generate output file path for a given file and model"""
    file_name = os.path.basename(file_path)
    model_name = model.replace("/", "_")
    return f"{dir_path}/output_test/{file_name}_{preset}_{model_name}.json"


def save_result_to_json(result_dict, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)


# Define main async entrypoint
async def main():
    # file_path = f"{dir_path}/2024-TRUE-annual_report.pdf"  ## local filepath and file URL supported
    # file_path = f"{dir_path}/moonling_th_word.pdf"
    file_path = f"{dir_path}/moonling_th_word_first_page.pdf"
    # file_path = f"{dir_path}/เปิดบัญชีกับ LX Phone V.pdf"
    # file_path = f"{dir_path}/วิธีการสมัครเป็น IB PC.V.pdf"
    # file_path = f"{dir_path}/test_wash_machine_100_rows_2.xlsx" # fail
    # file_path = (
    #     f"{dir_path}/AUDITOR_REPORT.DOCX"  ## local filepath and file URL supported
    # )
    # file_path = f"{dir_path}/Screenshot.png"

    ## process only some pages or all
    select_pages = (
        None  ## None for all, but could be int or list(int) page numbers (1 indexed)
    )

    output_dir = (
        f"{dir_path}/output_test"  ## directory to save the consolidated markdown file
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Process each model in the list
    for model in model_list:
        print(f"\n{'='*50}")
        print(f"Processing with model: {model}")
        print(f"{'='*50}")

        # Check if output file already exists
        output_file_path = get_output_file_path(file_path, model, preset)
        if os.path.exists(output_file_path):
            print(f"SKIP: Output file already exists: {output_file_path}")
            continue

        try:
            result: ZeroxOutput = await zerox(
                file_path=file_path,
                model=model,
                output_dir=output_dir,
                custom_system_prompt=custom_system_prompt,
                select_pages=select_pages,
                **kwargs,
            )

            # Convert result to json dict
            cost = calculate_cost(result.input_tokens, result.output_tokens, model)
            result_dict = {
                "completion_time": result.completion_time,
                "file_name": result.file_name,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": cost,
                "cost_thb": cost
                * 33,  # current usd to thb rate is 32.34 round up to 33
                "model_used": model,
                "pages": [
                    {
                        "content_length": page.content_length,
                        "content": page.content,
                        "page": page.page,
                    }
                    for page in result.pages
                ],
            }

            save_result_to_json(result_dict, output_file_path)
            print(f"✅ Completed: {output_file_path}")
            results.append(result_dict)

        except Exception as e:
            print(f"❌ Error processing with model {model}: {str(e)}")
            continue

    print(f"\n{'='*50}")
    print(f"Processing complete! Processed {len(results)} models.")
    print(f"{'='*50}")

    return results


# run the main function:
result = asyncio.run(main())
