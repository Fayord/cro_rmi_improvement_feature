import os
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime
import json
from dotenv import load_dotenv

# Define the metrics to validate with their descriptions
METRICS = {}


# Define the response schema for validation
class MetricValidation(BaseModel):
    reason: str = Field(description="Explanation of why the metric is valid or invalid")
    is_valid: bool = Field(
        description="Indicates if the text meets the specific risk assessment criteria for this metric"
    )


def add_metric(metric_name: str, metric_detail: str) -> Dict[str, str]:
    """Add a new metric with its description to the validation list."""
    global METRICS
    METRICS[metric_name] = metric_detail
    return METRICS


def remove_metric(metric_name: str) -> Dict[str, str]:
    """Remove a metric from the validation list."""
    global METRICS
    if metric_name in METRICS:
        del METRICS[metric_name]
    return METRICS


def get_metrics() -> Dict[str, str]:
    """Get the current list of metrics with their descriptions."""
    return METRICS


def get_llm(provider: str = "openai", model_name: Optional[str] = None):
    """Initialize an LLM based on the provider."""
    if provider == "openai":
        model = model_name if model_name else "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=0)
    elif provider == "local":
        model_path = model_name if model_name else "./models/gemma-7b.gguf"
        return LlamaCpp(
            model_path=model_path, temperature=0, max_tokens=2000, verbose=True
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def validate_text_on_metric(
    user_text: str,
    metric: str,
    metric_detail: str,
    additional_information: Optional[Dict[str, Any]] = None,
    provider: str = "openai",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate user text against a specific metric with detailed criteria and business context."""
    try:
        # Initialize LLM
        llm = get_llm(provider, model_name)

        # Setup validation parser
        parser = JsonOutputParser(pydantic_object=MetricValidation)

        # Create metric-specific criteria
        metric_criteria = f"""
        For this metric '{metric}':
        {metric_detail}
        """

        # Add business context if provided
        business_context = ""
        if additional_information:
            business_context = "\nBusiness Context:\n"
            for key, value in additional_information.items():
                business_context += f"- {key}: {value}\n"

        # Create prompt for this specific metric
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert risk assessment and validation specialist with deep expertise in evaluating risk management, control measures, and mitigation plans across Thai, Chinese, and English contexts. Your role is to provide thorough validation of risk-related content with cultural and linguistic sensitivity.

Evaluate if the following text meets the validation criteria, considering the provided business context:

{metric_criteria}
{business_context}

Return your assessment as a JSON with:
- 'is_valid': boolean (true/false)
- 'reason': string explaining your reasoning in English""",
                ),
                ("human", "Text to validate: {text}"),
            ]
        )

        # Create chain
        chain = prompt | llm | parser

        # Invoke the chain
        result = chain.invoke({"text": user_text})

        return {
            "text": user_text,
            "metric": metric,
            "metric_detail": metric_detail,
            "additional_information": additional_information,
            "is_valid": result.get("is_valid"),
            "reason": result.get("reason"),
            "provider": provider,
            "model": model_name or "default",
        }
    except Exception as e:
        return {
            "text": user_text,
            "metric": metric,
            "metric_detail": metric_detail,
            "additional_information": additional_information,
            "is_valid": False,
            "reason": f"Error during validation: {str(e)}",
            "provider": provider,
            "model": model_name or "default",
        }


def validate_text(
    user_text: str,
    metrics: Optional[List[str]] = None,
    provider: str = "openai",
    model_name: Optional[str] = None,
    additional_information: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate user text against all specified metrics."""
    metrics_to_use = list(metrics if metrics is not None else METRICS.keys())
    results = []

    for metric in metrics_to_use:
        print(f"Validating text on metric: {metric}")
        result = validate_text_on_metric(
            user_text,
            metric,
            METRICS[metric],
            additional_information=additional_information,
            provider=provider,
            model_name=model_name,
        )
        results.append(result)

    return results


def export_to_excel(
    results: List[Dict[str, Any]], filename: Optional[str] = None
) -> str:
    """
    Export validation results to Excel file.

    Args:
        results: List of validation results
        filename: Name of Excel file (optional)

    Returns:
        Path to the saved Excel file
    """
    df = pd.DataFrame(results)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provider = results[0]["provider"] if results else "unknown"
        filename = f"text_validation_{provider}_{timestamp}.xlsx"

    df.to_excel(filename, index=False)
    print(f"Results exported to {filename}")
    return filename


def create_human_review_template(
    results: List[Dict[str, Any]], filename: Optional[str] = None
) -> str:
    """
    Create Excel template for human review.

    Args:
        results: List of validation results
        filename: Name of Excel file (optional)

    Returns:
        Path to the saved Excel file
    """
    df = pd.DataFrame(results)

    # Add columns for human review
    df["human_is_valid"] = None
    df["human_reason"] = None

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provider = results[0]["provider"] if results else "unknown"
        filename = f"human_review_template_{provider}_{timestamp}.xlsx"

    df.to_excel(filename, index=False)
    print(f"Human review template exported to {filename}")
    return filename


def compare_with_human_review(
    llm_results_file: str, human_review_file: str, output_file: Optional[str] = None
) -> str:
    """
    Compare LLM validation results with human reviews.

    Args:
        llm_results_file: Path to LLM results Excel file
        human_review_file: Path to completed human review Excel file
        output_file: Path for comparison output Excel file (optional)

    Returns:
        Path to the comparison Excel file
    """
    # Load files
    llm_df = pd.read_excel(llm_results_file)
    human_df = pd.read_excel(human_review_file)

    # Merge on text and metric columns
    merged_df = pd.merge(
        llm_df,
        human_df[["text", "metric", "human_is_valid", "human_reason"]],
        on=["text", "metric"],
        how="left",
    )

    # Add comparison column
    merged_df["agreement"] = merged_df["is_valid"] == merged_df["human_is_valid"]

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llm_human_comparison_{timestamp}.xlsx"

    # Calculate summary statistics
    total = len(merged_df)
    agreed = merged_df["agreement"].sum()
    agreement_rate = agreed / total if total > 0 else 0

    # Metric-level agreement
    metric_agreement = merged_df.groupby("metric")["agreement"].agg(["count", "sum"])
    metric_agreement["rate"] = metric_agreement["sum"] / metric_agreement["count"]

    # Create Excel with multiple sheets
    with pd.ExcelWriter(output_file) as writer:
        merged_df.to_excel(writer, sheet_name="Detailed Comparison", index=False)

        summary_data = {
            "Metric": [
                "Total validations",
                "Agreed validations",
                "Overall agreement rate",
            ],
            "Value": [total, agreed, f"{agreement_rate:.2%}"],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        metric_agreement.to_excel(writer, sheet_name="Metric Agreement")

    print(f"Comparison exported to {output_file}")
    return output_file


class ImprovedExample(BaseModel):
    choice1: str = Field(description="First improved example of the risk description")
    choice2: str = Field(description="Second improved example of the risk description")
    choice3: str = Field(description="Third improved example of the risk description")


def _generate_improved_example(
    original_text: str,
    invalid_metrics: List[str],
) -> Dict[str, str]:
    """Generate improved examples based on the original text and invalid metrics using LLM.

    Args:
        original_text: The original user input text
        invalid_metrics: List of metrics that failed validation


    Returns:
        Dictionary containing three improved examples
    """
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=ImprovedExample)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert in risk assessment and validation. Generate three different improved versions of a risk description 
            that address the validation criteria that were not met in the original text.

            Each version should:
            1. Maintain the core risk context from the original text
            2. Address all the invalid metrics provided
            3. Be clear, concise, and specific
            4. Include cause, risk event, and impact where appropriate
            5. Be distinctly different from each other

            Return the results in JSON format with keys:
            - choice1: First improved version
            - choice2: Second improved version
            - choice3: Third improved version
            """,
            ),
            (
                "human",
                """Original text: {original_text}
            Failed metrics: {metrics}
            
            Please provide three improved versions that address these specific aspects.""",
            ),
        ]
    )

    chain = prompt | llm | parser

    result = chain.invoke(
        {
            "original_text": original_text,
            "metrics": ", ".join(invalid_metrics),
        }
    )

    return {
        "choice1": result.get("choice1", "No example sentence provided"),
        "choice2": result.get("choice2", "No example sentence provided"),
        "choice3": result.get("choice3", "No example sentence provided"),
    }


def generate_feedback_message(
    invalid_metrics: List[str], invalid_reasons: List[str], user_text: str
) -> str:
    """Generate a personalized feedback message using LLM."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert risk assessment advisor. Create a constructive and specific feedback message 
            about how to improve a risk description. Focus on the aspects that need improvement.""",
            ),
            (
                "human",
                """Original text: {text}
            Areas needing improvement: {metrics}
            Validation reasons: {reasons}
            
            Provide a concise, constructive feedback message explaining what needs to be improved and why. 
            and return with same language as Original text.""",
            ),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "text": user_text,
            "metrics": ", ".join(invalid_metrics),
            "reasons": ", ".join(invalid_reasons),
        }
    )

    return result.content


def generate_feedback_questions(
    validation_results: List[Dict[str, Any]], user_text: str
) -> Dict[str, Any]:
    """Generate a consolidated feedback question and examples for invalid metrics."""
    invalid_metrics = []
    invalid_reasons = []

    for result in validation_results:
        if not result["is_valid"]:
            invalid_metrics.append(result["metric"])
            invalid_reasons.append(result["reason"])

    if not invalid_metrics:
        return {}

    feedback_message = generate_feedback_message(
        invalid_metrics, invalid_reasons, user_text
    )

    # Generate improved examples using LLM
    examples = _generate_improved_example(user_text, invalid_metrics)

    return {
        "consolidated_feedback": {
            "invalid_metrics": invalid_metrics,
            "question": feedback_message,
            "examples": examples,
            "original_text": user_text,
        }
    }


# def _generate_improved_example(
#     original_text: str, invalid_metrics: List[str], variant: int
# ) -> str:
#     """Generate an improved example based on the original text and invalid metrics using LLM.

#     Args:
#         original_text: The original user input text
#         invalid_metrics: List of metrics that failed validation
#         variant: Integer to generate different variations of examples

#     Returns:
#         A string containing an improved example
#     """
#     llm = get_llm()

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are an expert in risk assessment and validation. Your task is to improve a given risk description
#         by addressing specific validation criteria that were not met in the original text. Generate an improved version that
#         specifically addresses these missing aspects while maintaining relevance to the original context.

#         The improved example should:
#         1. Maintain the core risk context from the original text
#         2. Address all the invalid metrics provided
#         3. Be clear, concise, and specific
#         4. Include cause, risk event, and impact where appropriate
#         5. Generate a different variation based on the variant number provided
#         """,
#             ),
#             (
#                 "human",
#                 """Original text: {original_text}
#         Failed metrics: {metrics}
#         Variation: {variant}

#         Please provide an improved version that addresses these specific aspects while maintaining the original context.
#         """,
#             ),
#         ]
#     )

#     chain = prompt | llm

#     result = chain.invoke(
#         {
#             "original_text": original_text,
#             "metrics": ", ".join(invalid_metrics),
#             "variant": f"Variation {variant}",
#         }
#     )

#     return result.content.strip()


# Main function to run the full flow
def validate_user_input(
    user_text: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
    export: bool = True,
    additional_information: Optional[Dict[str, Any]] = None,
):
    """Main function to validate user input text against all metrics.

    Args:
        user_text: The text to validate
        provider: LLM provider to use
        model_name: Name of the model to use
        export: Whether to export results to Excel
        additional_information: Optional dictionary containing business context

    Returns:
        Validation results, feedback, and optionally file paths
    """
    print(f"Validating user input: '{user_text}'")
    print(f"Using provider: {provider}, model: {model_name or 'default'}")
    print(f"Metrics: {METRICS}")

    # Validate text against all metrics
    results = validate_text(
        user_text,
        provider=provider,
        model_name=model_name,
        additional_information=additional_information,
    )

    # Generate consolidated feedback
    feedback = generate_feedback_questions(results, user_text)

    # Print results and feedback
    for result in results:
        valid_str = "VALID" if result["is_valid"] else "INVALID"
        print(f"\nMetric: {result['metric']} - {valid_str}")
        print(f"Reason: {result['reason']}")

    if feedback:
        print("\nConsolidated Feedback:")
        print(f"Question: {feedback['consolidated_feedback']['question']}")
        print("\nImprovement Examples:")
        for example in feedback["consolidated_feedback"]["examples"]:
            print(f"\n{example}")

    # Export if requested
    if export:
        file_path = export_to_excel(results)
        human_template = create_human_review_template(results)
        return results, feedback, file_path, human_template

    return results, feedback


# Example usage
if __name__ == "__main__":
    # Configure your API key
    env_path = "/Users/ford/Documents/coding/confidential/.env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "API key is missing"

    # Initialize default metrics
    add_metric(
        "Conciseness",
        "Text should be brief and to the point without unnecessary details",
    )

    add_metric(
        "Clarity",
        "Text should be easy to understand with precise language and clear risk descriptions",
    )
    add_metric(
        "Accuracy",
        "Text should contain factually correct information about risks and controls",
    )

    # Run with OpenAI
    user_input = input("Enter text to validate: ")
    results, file_path, template_path = validate_user_input(
        user_input, provider="openai"
    )

    # Optional: Run with a local model for comparison
    # local_results, local_file_path, local_template_path = validate_user_input(
    #     user_input,
    #     provider="local",
    #     model_name="./models/gemma-7b.gguf"
    # )

    print("\nAfter human review is completed, compare results with:")
    print(f"compare_with_human_review('{file_path}', 'completed_{template_path}')")
    print(results)
