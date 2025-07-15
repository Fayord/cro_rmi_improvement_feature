import os
from pyexpat import model
from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import
from langchain_core.messages import HumanMessage, SystemMessage
import io
from dotenv import load_dotenv
import base64

dir_path = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(dir_path, "../../../.env")
load_dotenv(env_path)


# --- Configuration ---
# Option 1: Using Google Cloud Project (Recommended for production)
# Ensure your GOOGLE_APPLICATION_CREDENTIALS are set up
# PROJECT_ID = "your-gcp-project-id" # Replace with your Google Cloud Project ID
# REGION = "us-central1" # Or another region where Gemini Flash is available
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash-latest",
#     project=PROJECT_ID,
#     location=REGION,
# )
# Option 2: Using a Google API Key (Simpler for development/quick tests)
# Get your API key from Google AI Studio: https://makersuite.google.com/app/apikey
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# model_name = "gemini-2.5-flash-lite-preview-06-17"
model_name = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(
    model=model_name,
)


# --- Function to extract OCR and format as Markdown ---
def extract_pdf_ocr_to_markdown(pdf_path: str) -> str:
    """
    Extracts OCR from a PDF using Gemini Flash via LangChain and returns markdown.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Encode the PDF content to base64
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # LangChain's way to pass file data directly to the model
    # The 'file_data' type is used for binary file content
    message_content = [
        {
            "type": "text",
            "text": "Extract all text from this document and format it as markdown, preserving structure, tables, and headings. Convert tables into markdown table format. Do not include any introductory or concluding remarks, just the extracted markdown content.",
        },
        # {
        #     "type": "file_data",
        #     "file_data": {"mime_type": "application/pdf", "data": pdf_bytes},
        # },
        {
            "type": "media",
            "mime_type": "application/pdf",
            "data": pdf_base64,
        },
    ]

    messages = [
        SystemMessage(
            content="You are an expert document transcriber and formatter. Your sole purpose is to convert provided document content into clean, structured markdown."
        ),
        HumanMessage(content=message_content),
    ]

    try:
        response = llm.invoke(messages)

        # Extract token usage and calculate costs
        input_tokens = response.usage_metadata["input_tokens"]
        output_tokens = response.usage_metadata["output_tokens"]

        # Calculate costs (pricing for gemini-2.5-flash-lite-preview-06-17)
        input_cost_per_1m = 0.10  # USD per 1M input tokens
        output_cost_per_1m = 0.40  # USD per 1M output tokens

        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        total_cost_usd = input_cost + output_cost
        total_cost_thb = total_cost_usd * 33  # USD to THB conversion

        print(f"\n💰 Token Usage Summary:")
        print(f"   Input tokens: {input_tokens:,}")
        print(f"   Output tokens: {output_tokens:,}")
        print(f"   Total tokens: {input_tokens + output_tokens:,}")
        print(f"   Input cost: ${input_cost:.6f}")
        print(f"   Output cost: ${output_cost:.6f}")
        print(f"   Total cost: ${total_cost_usd:.6f}")
        print(f"   Total cost: ฿{total_cost_thb:.6f}")

        return response.content
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return f"Error: Could not process PDF. {e}"


# --- Example Usage ---
if __name__ == "__main__":
    pdf_file_path = f"{dir_path}/moonling_th_word_first_page.pdf"
    # pdf_file_path = f"{dir_path}/2024-TRUE-annual_report.pdf"

    if os.path.exists(pdf_file_path):
        print(f"\nProcessing PDF: {pdf_file_path}")
        markdown_output = extract_pdf_ocr_to_markdown(pdf_file_path)
        print("\n--- Extracted Markdown ---")
        print(markdown_output)
        output_file_path = f"{dir_path}/output_test/output_direct_pdf_{model_name}.md"
        # Optional: Save to a markdown file
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        print(f"\nOutput saved to {output_file_path}")
    else:
        print(
            f"\nError: PDF file not found at {pdf_file_path}. Please provide a valid path."
        )
