"""
Simple PDF processing with Gemini 2.5 Flash using LiteLLM
Tracks input/output tokens and costs in USD and THB
"""

import os
from typing import Dict, Any
from litellm import completion
import json
from dotenv import load_dotenv
import time

# Exchange rate (33 THB to 1 USD)
THB_TO_USD_RATE = 33


def process_pdf_with_gemini(
    pdf_path: str,
    prompt: str,
    model: str,
) -> Dict[str, Any]:
    """
    Process a PDF file with Gemini 2.5 Flash and track costs.

    Args:
        pdf_path: Path to the PDF file
        prompt: Custom prompt for the model
        model: Model to use (default: gemini/gemini-2.5-flash)

    Returns:
        Dictionary containing response, token usage, and costs
    """

    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Make the API call with PDF input
        response = completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{encode_pdf_to_base64(pdf_path)}"
                            },
                        },
                    ],
                }
            ],
            # max_tokens=1000,
        )

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Calculate costs (approximate rates for Gemini 2.5 Flash)
        # Input: $0.000075 per 1K tokens, Output: $0.0003 per 1K tokens
        input_cost_usd = (input_tokens / 1000) * 0.000075
        output_cost_usd = (output_tokens / 1000) * 0.0003
        total_cost_usd = input_cost_usd + output_cost_usd

        # Convert to THB
        total_cost_thb = total_cost_usd * THB_TO_USD_RATE

        return {
            "response": response.choices[0].message.content,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
            },
            "costs": {
                "input_usd": round(input_cost_usd, 6),
                "output_usd": round(output_cost_usd, 6),
                "total_usd": round(total_cost_usd, 6),
                "total_thb": round(total_cost_thb, 6),
            },
            "model": model,
        }

    except Exception as e:
        return {
            "error": str(e),
            "tokens": {"input": 0, "output": 0, "total": 0},
            "costs": {"input_usd": 0, "output_usd": 0, "total_usd": 0, "total_thb": 0},
            "model": model,
        }


def encode_pdf_to_base64(pdf_path: str) -> str:
    """
    Encode PDF file to base64 string.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Base64 encoded string of the PDF
    """
    import base64

    with open(pdf_path, "rb") as pdf_file:
        pdf_data = pdf_file.read()
        return base64.b64encode(pdf_data).decode("utf-8")


def print_cost_summary(result: Dict[str, Any]) -> None:
    """
    Print a formatted cost and token summary.

    Args:
        result: Result dictionary from process_pdf_with_gemini
    """
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print("📊 Token and Cost Summary")
    print("=" * 40)
    print(f"Model: {result['model']}")
    print(f"Input tokens: {result['tokens']['input']:,}")
    print(f"Output tokens: {result['tokens']['output']:,}")
    print(f"Total tokens: {result['tokens']['total']:,}")
    print()
    print("💰 Costs:")
    print(f"  Input cost: ${result['costs']['input_usd']:.6f}")
    print(f"  Output cost: ${result['costs']['output_usd']:.6f}")
    print(f"  Total cost: ${result['costs']['total_usd']:.6f}")
    print(f"  Total cost: ฿{result['costs']['total_thb']:.2f}")
    print()
    print("📄 Response Preview:")
    response = result["response"]
    print(response[:200] + "..." if len(response) > 200 else response)


def main():
    """
    Example usage of the PDF processor.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(dir_path, "../../../.env")
    load_dotenv(env_path)

    # Example usage
    # pdf_path = "example.pdf"  # Replace with your PDF path
    # pdf_path = f"{dir_path}/moonling_th_word_first_page.pdf"
    pdf_path = f"{dir_path}/2024-TRUE-annual_report.pdf"

    prompt = "Extract all text from this document and format it as markdown, preserving structure, tables, and headings. Convert tables into markdown table format. Do not include any introductory or concluding remarks, just the extracted markdown content."
    print("🚀 Processing PDF with Gemini 2.5 Flash...")
    print(f"PDF: {pdf_path}")
    print(f"Prompt: {prompt}")
    print()
    # time start
    start_time = time.time()
    model = "gemini/gemini-2.5-flash"
    result = process_pdf_with_gemini(pdf_path, prompt, model)
    # time end
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print_cost_summary(result)
    # save result to json file
    output_file_path = f"{dir_path}/output_test/output_direct_pdf_litellm_{model.replace('/', '_')}.json"
    with open(output_file_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved to {output_file_path}")


if __name__ == "__main__":
    main()
