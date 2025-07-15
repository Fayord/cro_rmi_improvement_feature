"""
Concurrent PDF processing with Gemini 2.5 Flash using LiteLLM
Splits PDF into pages, processes each page concurrently, and merges results
Tracks input/output tokens and costs in USD and THB
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from litellm import completion
from dotenv import load_dotenv
import base64
from PyPDF2 import PdfReader
import io
import time
import psutil
import gc

# Exchange rate (33 THB to 1 USD)
THB_TO_USD_RATE = 33


@dataclass
class PageData:
    """Data structure for a single page result"""

    page_number: int
    page_content: str
    tokens: Dict[str, int]
    costs: Dict[str, float]
    error: Optional[str] = None


def extract_pdf_pages_ultra_optimized(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract individual pages from PDF with minimal overhead.
    Pre-processes all pages in one pass to avoid repeated PDF operations.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing page data and base64 content
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []

    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            total_pages = len(pdf_reader.pages)

            print(f"📄 Pre-processing {total_pages} pages (one-time operation)...")

            # Pre-process all pages in one pass
            for page_num in range(total_pages):
                # Create a new PDF with just this page
                from PyPDF2 import PdfWriter

                writer = PdfWriter()
                writer.add_page(pdf_reader.pages[page_num])

                # Convert to base64 (this is the expensive operation)
                output_stream = io.BytesIO()
                writer.write(output_stream)
                output_stream.seek(0)
                page_base64 = base64.b64encode(output_stream.getvalue()).decode("utf-8")

                pages.append(
                    {
                        "page_number": page_num + 1,
                        "base64_content": page_base64,
                        "total_pages": total_pages,
                    }
                )

                # Progress indicator for long PDFs
                if total_pages > 10 and (page_num + 1) % 5 == 0:
                    print(f"   Pre-processed {page_num + 1}/{total_pages} pages")

    except Exception as e:
        raise Exception(f"Error extracting PDF pages: {str(e)}")

    return pages


def process_single_page_optimized(
    page_data: Dict[str, Any], prompt: str, model: str
) -> PageData:
    """
    Process a single PDF page with Gemini 2.5 Flash (optimized version).

    Args:
        page_data: Dictionary containing page information and base64 content
        prompt: Custom prompt for the model
        model: Model to use

    Returns:
        PageData object with results
    """
    page_number = page_data["page_number"]
    base64_content = page_data["base64_content"]
    total_pages = page_data["total_pages"]

    # Customize prompt for individual page
    page_prompt = f"{prompt}\n\nThis is page {page_number} of {total_pages}."

    try:
        # Make the API call with PDF page input (no artificial delays)
        response = completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": page_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{base64_content}"
                            },
                        },
                    ],
                }
            ],
            reasoning_effort="disable",
        )

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Calculate costs (approximate rates for Gemini 2.5 Flash)
        input_cost_usd = (input_tokens / 1000) * 0.000075
        output_cost_usd = (output_tokens / 1000) * 0.0003
        total_cost_usd = input_cost_usd + output_cost_usd
        total_cost_thb = total_cost_usd * THB_TO_USD_RATE

        return PageData(
            page_number=page_number,
            page_content=response.choices[0].message.content,
            tokens={
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
            },
            costs={
                "input_usd": round(input_cost_usd, 6),
                "output_usd": round(output_cost_usd, 6),
                "total_usd": round(total_cost_usd, 6),
                "total_thb": round(total_cost_thb, 6),
            },
        )

    except Exception as e:
        return PageData(
            page_number=page_number,
            page_content="",
            tokens={"input": 0, "output": 0, "total": 0},
            costs={"input_usd": 0, "output_usd": 0, "total_usd": 0, "total_thb": 0},
            error=str(e),
        )


def process_pdf_concurrent_ultra_optimized(
    pdf_path: str,
    prompt: str,
    model: str = "gemini/gemini-2.5-flash",
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    Process a PDF file concurrently with ultra-optimized approach.

    Args:
        pdf_path: Path to the PDF file
        prompt: Custom prompt for the model
        model: Model to use
        max_workers: Maximum number of concurrent workers

    Returns:
        Dictionary containing merged results, token usage, and costs
    """

    # Pre-process all pages (one-time operation)
    print(f"📄 Pre-processing PDF pages from {pdf_path}...")
    start_preprocess = time.time()
    pages = extract_pdf_pages_ultra_optimized(pdf_path)
    preprocess_time = time.time() - start_preprocess
    print(f"✅ Pre-processed {len(pages)} pages in {preprocess_time:.2f}s")

    # Process pages concurrently (API calls only)
    print(
        f"🚀 Processing {len(pages)} pages concurrently with {max_workers} workers..."
    )
    start_api_calls = time.time()
    page_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(process_single_page_optimized, page, prompt, model): page
            for page in pages
        }

        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_data = future_to_page[future]
            try:
                result = future.result()
                page_results.append(result)
                print(f"✅ Page {result.page_number} completed")
            except Exception as e:
                print(f"❌ Page {page_data['page_number']} failed: {str(e)}")
                # Add error result
                page_results.append(
                    PageData(
                        page_number=page_data["page_number"],
                        page_content="",
                        tokens={"input": 0, "output": 0, "total": 0},
                        costs={
                            "input_usd": 0,
                            "output_usd": 0,
                            "total_usd": 0,
                            "total_thb": 0,
                        },
                        error=str(e),
                    )
                )

    api_calls_time = time.time() - start_api_calls
    print(f"⏱️  API calls completed in {api_calls_time:.2f}s")

    # Sort results by page number
    page_results.sort(key=lambda x: x.page_number)

    # Calculate totals
    total_input_tokens = sum(r.tokens["input"] for r in page_results)
    total_output_tokens = sum(r.tokens["output"] for r in page_results)
    total_tokens = sum(r.tokens["total"] for r in page_results)

    total_input_cost_usd = sum(r.costs["input_usd"] for r in page_results)
    total_output_cost_usd = sum(r.costs["output_usd"] for r in page_results)
    total_cost_usd = sum(r.costs["total_usd"] for r in page_results)
    total_cost_thb = sum(r.costs["total_thb"] for r in page_results)

    # Count errors
    error_count = sum(1 for r in page_results if r.error)

    return {
        "pages": [
            {
                "page_number": r.page_number,
                "page_content": r.page_content,
                "tokens": r.tokens,
                "costs": r.costs,
                "error": r.error,
            }
            for r in page_results
        ],
        "summary": {
            "total_pages": len(page_results),
            "successful_pages": len(page_results) - error_count,
            "failed_pages": error_count,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_tokens,
            },
            "costs": {
                "input_usd": round(total_input_cost_usd, 6),
                "output_usd": round(total_output_cost_usd, 6),
                "total_usd": round(total_cost_usd, 6),
                "total_thb": round(total_cost_thb, 6),
            },
            "model": model,
            "timing": {
                "preprocess_time": round(preprocess_time, 2),
                "api_calls_time": round(api_calls_time, 2),
                "total_time": round(preprocess_time + api_calls_time, 2),
            },
        },
    }


def print_concurrent_summary(result: Dict[str, Any]) -> None:
    """
    Print a formatted summary of concurrent processing results.

    Args:
        result: Result dictionary from process_pdf_concurrent
    """
    summary = result["summary"]

    print("📊 Concurrent Processing Summary")
    print("=" * 50)
    print(f"Model: {summary['model']}")
    print(f"Total pages: {summary['total_pages']}")
    print(f"Successful pages: {summary['successful_pages']}")
    print(f"Failed pages: {summary['failed_pages']}")
    print()
    print("💰 Total Costs:")
    print(f"  Input cost: ${summary['costs']['input_usd']:.6f}")
    print(f"  Output cost: ${summary['costs']['output_usd']:.6f}")
    print(f"  Total cost: ${summary['costs']['total_usd']:.6f}")
    print(f"  Total cost: ฿{summary['costs']['total_thb']:.2f}")
    print()
    print("📄 Token Usage:")
    print(f"  Input tokens: {summary['tokens']['input']:,}")
    print(f"  Output tokens: {summary['tokens']['output']:,}")
    print(f"  Total tokens: {summary['tokens']['total']:,}")
    print()

    # Show timing breakdown if available
    if "timing" in summary:
        print("⏱️  Timing Breakdown:")
        print(f"  Pre-processing: {summary['timing']['preprocess_time']}s")
        print(f"  API calls: {summary['timing']['api_calls_time']}s")
        print(f"  Total: {summary['timing']['total_time']}s")
        print()

    # Show memory management info if available
    if "memory_management" in summary:
        print("🐳 Memory Management:")
        print(f"  Initial workers: {summary['memory_management']['initial_workers']}")
        print(
            f"  Final memory: {summary['memory_management']['final_memory_percent']}%"
        )
        print(
            f"  Memory threshold: {summary['memory_management']['memory_threshold']}%"
        )
        print()

    # Show page-by-page results
    print("📋 Page Results:")
    for page in result["pages"]:
        status = "✅" if not page["error"] else "❌"
        print(
            f"  {status} Page {page['page_number']}: {page['tokens']['total']} tokens, ${page['costs']['total_usd']:.6f}"
        )
        if page["error"]:
            print(f"    Error: {page['error']}")


def compare_processing_methods():
    """
    Compare single process vs concurrent processing methods.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(dir_path, "../../../.env")
    load_dotenv(env_path)

    pdf_path = f"{dir_path}/2024-TRUE-annual_report.pdf"
    prompt = "Extract all text from this document and format it as markdown, preserving structure, tables, and headings. Convert tables into markdown table format. Do not include any introductory or concluding remarks, just the extracted markdown content."
    model = "gemini/gemini-2.5-flash"

    print("🔍 Comparing Processing Methods")
    print("=" * 50)
    print(f"PDF: {pdf_path}")
    print(f"Model: {model}")
    print()

    # Test 1: Single process (from litellm_direct_pdf.py)
    print("📊 Test 1: Single Process")
    print("-" * 30)
    start_time = time.time()

    # Import the single process function
    import sys

    sys.path.append(dir_path)
    from litellm_direct_pdf import process_pdf_with_gemini

    single_result = process_pdf_with_gemini(pdf_path, prompt, model)
    single_time = time.time() - start_time
    print(f"⏱️  Single process time: {single_time:.2f} seconds")
    print()

    # Test 2: Concurrent process (optimized)
    print("📊 Test 2: Concurrent Process (Optimized)")
    print("-" * 40)
    start_time = time.time()
    concurrent_result = process_pdf_concurrent_ultra_optimized(
        pdf_path, prompt, model, max_workers=4
    )
    concurrent_time = time.time() - start_time
    print(f"⏱️  Concurrent process time: {concurrent_time:.2f} seconds")
    print()

    # Comparison
    print("📈 Performance Comparison")
    print("=" * 30)
    speedup = single_time / concurrent_time if concurrent_time > 0 else 0
    print(f"Single process: {single_time:.2f}s")
    print(f"Concurrent process: {concurrent_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1:
        print("✅ Concurrent processing is faster!")
    elif speedup < 1:
        print("❌ Single processing is faster (likely due to API rate limits)")
    else:
        print("⚖️  Both methods have similar performance")


def test_optimal_workers(
    pdf_path: str,
    prompt: str,
    model: str = "gemini/gemini-2.5-flash",
    max_workers_list: List[int] = [2, 4, 6, 8],
) -> Dict[str, Any]:
    """
    Test different worker counts to find optimal performance.

    Args:
        pdf_path: Path to the PDF file
        prompt: Custom prompt for the model
        model: Model to use
        max_workers_list: List of worker counts to test

    Returns:
        Dictionary with performance results for each worker count
    """
    print("🔬 Testing Optimal Worker Count")
    print("=" * 50)
    print(f"PDF: {pdf_path}")
    print(f"Model: {model}")
    print()

    results = {}

    for workers in max_workers_list:
        print(f"🧪 Testing with {workers} workers...")
        start_time = time.time()

        try:
            result = process_pdf_concurrent_ultra_optimized(
                pdf_path, prompt, model, max_workers=workers
            )

            total_time = time.time() - start_time
            api_time = result["summary"]["timing"]["api_calls_time"]
            preprocess_time = result["summary"]["timing"]["preprocess_time"]

            results[workers] = {
                "total_time": total_time,
                "api_time": api_time,
                "preprocess_time": preprocess_time,
                "successful_pages": result["summary"]["successful_pages"],
                "failed_pages": result["summary"]["failed_pages"],
                "total_cost_usd": result["summary"]["costs"]["total_usd"],
            }

            print(
                f"✅ {workers} workers: {total_time:.2f}s total, {api_time:.2f}s API calls"
            )

        except Exception as e:
            print(f"❌ {workers} workers failed: {str(e)}")
            results[workers] = {"error": str(e)}

    # Find optimal configuration
    successful_results = {k: v for k, v in results.items() if "error" not in v}

    if successful_results:
        fastest_workers = min(
            successful_results.keys(), key=lambda w: successful_results[w]["total_time"]
        )
        fastest_time = successful_results[fastest_workers]["total_time"]

        print("\n📊 Performance Summary:")
        print("=" * 30)
        for workers, data in successful_results.items():
            speedup = fastest_time / data["total_time"]
            print(f"  {workers} workers: {data['total_time']:.2f}s ({speedup:.2f}x)")

        print(f"\n🏆 Optimal: {fastest_workers} workers ({fastest_time:.2f}s)")

    return results


def get_optimal_workers(pdf_pages: int, api_rate_limit: int = 60) -> int:
    """
    Calculate optimal worker count based on PDF size and API limits.

    Args:
        pdf_pages: Number of pages in PDF
        api_rate_limit: API requests per minute (default: 60)

    Returns:
        Recommended worker count
    """
    # Conservative approach: don't exceed 80% of rate limit
    max_safe_workers = int(api_rate_limit * 0.8 / 60)  # requests per second

    # Adjust based on PDF size
    if pdf_pages <= 5:
        return min(2, pdf_pages)
    elif pdf_pages <= 20:
        return min(4, max_safe_workers)
    elif pdf_pages <= 50:
        return min(6, max_safe_workers)
    else:
        return min(8, max_safe_workers)


def get_memory_usage() -> float:
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent


def get_safe_workers_for_memory(
    pdf_pages: int, target_memory_percent: float = 80
) -> int:
    """
    Calculate safe number of workers based on current memory usage.

    Args:
        pdf_pages: Number of pages in PDF
        target_memory_percent: Target memory usage (default: 80%)

    Returns:
        Safe number of workers
    """
    current_memory = get_memory_usage()

    # If memory is already high, reduce workers
    if current_memory > target_memory_percent:
        # Reduce workers by memory pressure
        memory_pressure = current_memory / target_memory_percent
        base_workers = min(4, pdf_pages)
        safe_workers = max(1, int(base_workers / memory_pressure))
        print(
            f"⚠️  High memory usage ({current_memory:.1f}%), reducing workers to {safe_workers}"
        )
        return safe_workers

    # Normal calculation
    if pdf_pages <= 10:
        return min(4, pdf_pages)
    elif pdf_pages <= 50:
        return min(6, pdf_pages)
    else:
        return min(8, pdf_pages)


def process_pdf_with_memory_management(
    pdf_path: str,
    prompt: str,
    model: str = "gemini/gemini-2.5-flash",
    max_workers: int = 8,
    memory_threshold: float = 85.0,
) -> Dict[str, Any]:
    """
    Process PDF with automatic memory management.

    Args:
        pdf_path: Path to the PDF file
        prompt: Custom prompt for the model
        model: Model to use
        max_workers: Maximum workers to use
        memory_threshold: Memory percentage threshold (default: 85%)

    Returns:
        Dictionary containing results with memory management info
    """

    # Pre-process all pages
    print(f"📄 Pre-processing PDF pages from {pdf_path}...")
    start_preprocess = time.time()
    pages = extract_pdf_pages_ultra_optimized(pdf_path)
    preprocess_time = time.time() - start_preprocess
    print(f"✅ Pre-processed {len(pages)} pages in {preprocess_time:.2f}s")

    # Calculate initial safe workers
    initial_workers = min(max_workers, get_safe_workers_for_memory(len(pages)))
    print(
        f"📊 Starting with {initial_workers} workers (memory: {get_memory_usage():.1f}%)"
    )

    # Process pages with memory monitoring
    print(f"🚀 Processing {len(pages)} pages with memory management...")
    start_api_calls = time.time()
    page_results = []

    # Process in batches to monitor memory
    batch_size = 10  # Process 10 pages at a time
    current_workers = initial_workers

    for batch_start in range(0, len(pages), batch_size):
        batch_end = min(batch_start + batch_size, len(pages))
        batch_pages = pages[batch_start:batch_end]

        print(
            f"📦 Processing batch {batch_start//batch_size + 1}: pages {batch_start+1}-{batch_end}"
        )
        print(
            f"   Current memory: {get_memory_usage():.1f}%, workers: {current_workers}"
        )

        # Check memory before each batch
        if get_memory_usage() > memory_threshold:
            # Reduce workers if memory is high
            current_workers = max(1, current_workers - 1)
            print(f"⚠️  Memory high, reducing workers to {current_workers}")

            # Force garbage collection
            gc.collect()

        # Process batch
        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            future_to_page = {
                executor.submit(
                    process_single_page_optimized, page, prompt, model
                ): page
                for page in batch_pages
            }

            for future in as_completed(future_to_page):
                page_data = future_to_page[future]
                try:
                    result = future.result()
                    page_results.append(result)
                    print(f"✅ Page {result.page_number} completed")
                except Exception as e:
                    print(f"❌ Page {page_data['page_number']} failed: {str(e)}")
                    page_results.append(
                        PageData(
                            page_number=page_data["page_number"],
                            page_content="",
                            tokens={"input": 0, "output": 0, "total": 0},
                            costs={
                                "input_usd": 0,
                                "output_usd": 0,
                                "total_usd": 0,
                                "total_thb": 0,
                            },
                            error=str(e),
                        )
                    )

        # Memory cleanup after each batch
        gc.collect()

        # Adjust workers for next batch if needed
        if get_memory_usage() < memory_threshold - 10:  # Memory is low
            current_workers = min(max_workers, current_workers + 1)
            print(f"✅ Memory low, increasing workers to {current_workers}")

    api_calls_time = time.time() - start_api_calls
    print(f"⏱️  API calls completed in {api_calls_time:.2f}s")
    print(f"📊 Final memory usage: {get_memory_usage():.1f}%")

    # Sort results by page number
    page_results.sort(key=lambda x: x.page_number)

    # Calculate totals
    total_input_tokens = sum(r.tokens["input"] for r in page_results)
    total_output_tokens = sum(r.tokens["output"] for r in page_results)
    total_tokens = sum(r.tokens["total"] for r in page_results)

    total_input_cost_usd = sum(r.costs["input_usd"] for r in page_results)
    total_output_cost_usd = sum(r.costs["output_usd"] for r in page_results)
    total_cost_usd = sum(r.costs["total_usd"] for r in page_results)
    total_cost_thb = sum(r.costs["total_thb"] for r in page_results)

    # Count errors
    error_count = sum(1 for r in page_results if r.error)

    return {
        "pages": [
            {
                "page_number": r.page_number,
                "page_content": r.page_content,
                "tokens": r.tokens,
                "costs": r.costs,
                "error": r.error,
            }
            for r in page_results
        ],
        "summary": {
            "total_pages": len(page_results),
            "successful_pages": len(page_results) - error_count,
            "failed_pages": error_count,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_tokens,
            },
            "costs": {
                "input_usd": round(total_input_cost_usd, 6),
                "output_usd": round(total_output_cost_usd, 6),
                "total_usd": round(total_cost_usd, 6),
                "total_thb": round(total_cost_thb, 6),
            },
            "model": model,
            "timing": {
                "preprocess_time": round(preprocess_time, 2),
                "api_calls_time": round(api_calls_time, 2),
                "total_time": round(preprocess_time + api_calls_time, 2),
            },
            "memory_management": {
                "initial_workers": initial_workers,
                "final_memory_percent": round(get_memory_usage(), 1),
                "memory_threshold": memory_threshold,
            },
        },
    }


def main():
    """
    Example usage of the concurrent PDF processor with memory management.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(dir_path, "../../../.env")
    load_dotenv(env_path)

    # Example usage
    # pdf_path = f"{dir_path}/moonling_th_word.pdf"
    pdf_path = f"{dir_path}/2024-TRUE-annual_report.pdf"

    prompt = "Extract all text from this document and format it as markdown, preserving structure, tables, and headings. Convert tables into markdown table format. Do not include any introductory or concluding remarks, just the extracted markdown content."

    print("🚀 Processing PDF with memory management for Docker...")
    print(f"PDF: {pdf_path}")
    print(f"Prompt: {prompt}")
    print()
    # time start
    start_time = time.time()
    model = "gemini/gemini-2.5-flash"

    # Use memory-managed version for Docker safety
    print("🐳 Using memory-managed processing for Docker environment")
    result = process_pdf_with_memory_management(
        pdf_path, prompt, model, max_workers=6, memory_threshold=85.0
    )
    # time end
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print_concurrent_summary(result)

    # Save result to JSON file
    output_file_path = f"{dir_path}/output_test/output_memory_managed_pdf_litellm_{model.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved to {output_file_path}")


def test_optimal():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(dir_path, "../../../.env")
    load_dotenv(env_path)

    pdf_path = f"{dir_path}/2024-TRUE-annual_report.pdf"
    prompt = "Extract all text from this document and format it as markdown, preserving structure, tables, and headings. Convert tables into markdown table format. Do not include any introductory or concluding remarks, just the extracted markdown content."
    model = "gemini/gemini-2.5-flash"

    result = test_optimal_workers(pdf_path, prompt, model)

    # Save result to JSON file
    output_file_path = f"{dir_path}/output_test/output_concurrent_pdf_litellm_{model.replace('/', '_')}_test_optimal.json"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved to {output_file_path}")


if __name__ == "__main__":
    # Uncomment to run comparison
    # compare_processing_methods()

    # Run normal processing
    main()
    # test_optimal()
