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
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

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

    return df


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
            4. Be distinctly different from each other

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
            
            Please provide three improved versions sentences that address these specific aspects.""",
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


def generate_improved_example_text(
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

    # Generate improved examples using LLM
    examples = _generate_improved_example(user_text, invalid_metrics)

    return {
        "improved_example_text": {
            "invalid_metrics": invalid_metrics,
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
    additional_information: Optional[Dict[str, Any]] = None,
):
    """Main function to validate user input text against all metrics.

    Args:
        user_text: The text to validate
        provider: LLM provider to use
        model_name: Name of the model to use
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
    improved_example_text = generate_improved_example_text(results, user_text)
    metrics_assessment = []
    for result in results:
        if not result["is_valid"]:
            metrics_assessment.append(
                {
                    "metric_name": result["metric"],
                    "metric_description": result["metric_detail"],
                    "reason": result["reason"],
                }
            )

    feedback = generate_executive_summary_feedback(
        user_sentence=user_text, metrics_assessment=metrics_assessment
    )

    return results, improved_example_text, feedback


from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# Define the Pydantic models for structured output
class Reason(BaseModel):
    reason_title: str = Field(description="A short title for the reason")
    reason_description: str = Field(
        description="Detailed explanation of why this is an issue"
    )
    related_metrics: List[str] = Field(
        description="List of metrics related to this reason"
    )
    improvement_suggestion: str = Field(description="Suggestion on how to improve")


class ExecutiveSummary(BaseModel):
    summary_paragraph: str = Field(
        description="A concise executive summary paragraph explaining overall assessment"
    )
    reasons: List[Reason] = Field(
        description="List of reasons why the user sentence did not pass metrics, maximum 5 reasons",
        max_items=5,
    )


def generate_executive_summary_feedback(
    user_sentence: str,
    metrics_assessment: List[Dict],
    model_name: str = "gpt-4o-mini",
) -> ExecutiveSummary:
    """
    Generate an executive summary with reasons why a user's sentence didn't pass certain metrics.
    Reasons can be consolidated from multiple metrics.

    Args:
        user_sentence: The sentence provided by the user
        metrics_assessment: List of dictionaries with metric_name, metric_description, and reason
        api_key: OpenAI API key
        model_name: The model to use (default: gpt-4o-mini)

    Returns:
        ExecutiveSummary object with summary_paragraph and reasons (max 5)
    """
    if metrics_assessment == []:
        return {}
    api_key = os.getenv("OPENAI_API_KEY", None)
    assert api_key is not None, "OPENAI_API_KEY is not set"
    # Initialize the LLM
    llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.1)

    # Create a Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=ExecutiveSummary)

    # Create a prompt template
    template = """
    You are an expert content analyzer. Your task is to create an executive summary explaining 
    why a user's sentence did not pass certain metrics and provide reasons for improvement.
    
    User sentence: {user_sentence}
    
    Metrics assessment:
    {metrics_assessment}
    
    Create an executive summary with:
    1. A single concise summary paragraph that captures the overall assessment
    2. A maximum of 5 key reasons why the sentence didn't pass metrics
    
    IMPORTANT: Each reason should be a consolidated theme that may relate to multiple metrics.
    Don't create a separate reason for each metric - instead, group related issues together.
    For example, if there are issues with both "Clarity" and "Specificity" metrics, these could
    be combined into a single reason about "Lack of Clear and Specific Details".

    For each reason, include:
    - A short, clear reason title
    - A detailed explanation of the issue
    - A list of the related metrics that this reason addresses
    - A concrete suggestion for improvement
    
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the chain
    chain = prompt | llm | parser

    # Execute the chain
    result = chain.invoke(
        {"user_sentence": user_sentence, "metrics_assessment": metrics_assessment}
    )
    result = result.model_dump(mode="json")
    result["original_text"] = user_sentence
    return result


# Example usage
if __name__ == "__main__":
    # Configure your API key
    env_path = "/Users/ford/Documents/coding/confidential/.env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "API key is missing"

    # # Initialize default metrics
    # add_metric(
    #     "Conciseness",
    #     "Text should be brief and to the point without unnecessary details",
    # )

    # add_metric(
    #     "Clarity",
    #     "Text should be easy to understand with precise language and clear risk descriptions",
    # )
    # add_metric(
    #     "Accuracy",
    #     "Text should contain factually correct information about risks and controls",
    # )

    # # Run with OpenAI
    # user_input = input("Enter text to validate: ")
    # results, file_path, template_path = validate_user_input(
    #     user_input, provider="openai"
    # )

    # # Optional: Run with a local model for comparison
    # # local_results, local_file_path, local_template_path = validate_user_input(
    # #     user_input,
    # #     provider="local",
    # #     model_name="./models/gemma-7b.gguf"
    # # )

    # print("\nAfter human review is completed, compare results with:")
    # print(f"compare_with_human_review('{file_path}', 'completed_{template_path}')")
    # print(results)
    user_sentence = "The product is good."

    # Example with more than 5 metrics
    metrics_assessment = [
        {
            "metric_name": "Specificity",
            "metric_description": "Sentence should provide specific details about the subject",
            "reason": "The statement is too vague and doesn't specify what aspects make the product good",
        },
        {
            "metric_name": "Evidence",
            "metric_description": "Claims should be supported with evidence or examples",
            "reason": "No supporting evidence or examples are provided to back up the claim",
        },
        {
            "metric_name": "Length",
            "metric_description": "Response should be at least 15 words",
            "reason": "The sentence contains only 4 words, which is below the minimum requirement",
        },
        {
            "metric_name": "Audience Targeting",
            "metric_description": "Content should address the intended audience",
            "reason": "The statement doesn't consider the target audience's interests or needs",
        },
        {
            "metric_name": "Comparison",
            "metric_description": "Content should provide comparative analysis when relevant",
            "reason": "No comparison to alternatives or previous versions is provided",
        },
        {
            "metric_name": "Clarity",
            "metric_description": "Statement should be clear and easy to understand",
            "reason": "While simple, the statement lacks clarity about what 'good' means in this context",
        },
        {
            "metric_name": "Actionability",
            "metric_description": "Content should provide actionable insights",
            "reason": "The statement provides no actionable information for the reader",
        },
    ]

    # Replace with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    executive_summary = generate_executive_summary_feedback(
        user_sentence=user_sentence,
        metrics_assessment=metrics_assessment,
    )
    executive_summary_dict = executive_summary.model_dump(mode="json")
    print(executive_summary_dict)
    # print(f"Summary: {executive_summary.summary_paragraph}")
    # print("\nReasons:")
    # for i, reason in enumerate(executive_summary.reasons, 1):
    #     print(f"{i}. {reason.reason_title}")
    #     print(f"   {reason.reason_description}")
    #     print(f"   Related metrics: {', '.join(reason.related_metrics)}")
    #     print(f"   Suggestion: {reason.improvement_suggestion}")
    #     print()
