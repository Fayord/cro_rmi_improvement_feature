# The granularity_classifier is designed to programmatically assess and categorize input text into a multi-level depth model, providing classifications at varying degrees of specificity.

# Purpose: To classify a given phrase or sentence into one of four defined granularity levels:

# "too specific": Captures atomic expressions or distinct factoids, providing maximum detail.

# "specific": Denotes specific events or concepts with clear cause/type.

# "general": Denotes broader categories without specific causes.

# "too general": Represents the overarching topic or domain, which must always be populated.

# Conditional Output: A key feature is its ability to output to be "too specific", "specific", "general", "too general"
# for "too general" we have set of data to use as a reference

# I want to create this for risk assessment, the topic is risk, control, rootcause, process
# use langchain to create this

import os
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Set up caching

dir_path = os.path.dirname(os.path.realpath(__file__))
set_llm_cache(SQLiteCache(database_path=f"{dir_path}/.langchain.db"))

# Load environment variables
try:
    env_path = f"{dir_path}/../../../../.env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "API key is missing"
except Exception:
    # Fallback path
    env_path = f"{dir_path}/../../../../../coding/confidential/.env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "API key is missing"


class GranularityClassification(BaseModel):
    """Structured output for granularity classification."""

    level: Literal["too specific", "specific", "general", "too general"] = Field(
        description="Granularity level: too specific, specific, general, too general"
    )

    confidence: int = Field(
        ge=1, le=5, description="Confidence level from 1 (low) to 5 (high)"
    )

    reasoning: str = Field(description="Brief explanation of why this level was chosen")


def get_llm(provider: str = "openai", model_name: Optional[str] = None) -> ChatOpenAI:
    """Initialize an LLM based on the provider."""
    if provider == "openai":
        model = model_name if model_name else "gpt-4.1-mini"
        temperature = 0.1 if model_name not in ["o3-mini"] else None
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def classify_granularity(
    text: str,
    context: Literal["risk", "control", "rootcause", "process"],
    description: str,
    reference_categories: Optional[List[str]] = None,
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Classify the granularity level of input text for risk assessment based on context domain.

    Args:
        text: The text to classify (this is what gets classified for granularity)
        context: The domain context ("risk", "control", "rootcause", "process")
        description: Additional context that provides more specific details to help with classification
        reference_categories: Optional list of reference categories for "too general" classification
        model_name: The LLM model to use

    Returns:
        Dictionary containing classification results
    """

    try:
        llm = get_llm(model_name=model_name)

        # Define domain-specific prompts
        domain_prompts = {
            "risk": {
                "system": """You are an expert in risk assessment and business analysis. Your task is to classify the granularity level of input text specifically related to RISK identification and assessment.

GRANULARITY LEVELS:
- "too specific": Captures atomic expressions or distinct factoids with maximum detail
  Examples: "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM", "Fire broke out in Building A, Floor 3, Room 305 at 3:45 PM on January 15th"

- "specific": Denotes specific risk events or scenarios with clear cause/type
  Examples: "Business interruption from fire hazards", "Business interruption from flood", "Phishing attack targeting employees", "Data breach from unauthorized access", "Supply chain disruption from supplier bankruptcy"

- "general": Denotes broader risk categories without specific causes
  Examples: "Business interruption", "Cybersecurity breach", "Supply chain disruption", "Operational disruption", "Financial loss"

- "too general": Represents the overarching topic or domain
  Examples: "Operational Risk", "Cybersecurity Risk", "Financial Risk", "Strategic Risk", "External Threats"

RISK DOMAIN CONTEXT:
- Focus on potential events, threats, vulnerabilities, or exposures that could cause harm
- Consider likelihood, impact, and severity of the risk
- Look for specific risk events, risk scenarios, or risk factors
- For "too general", categorize into broad risk types (e.g., Operational Risk, Financial Risk, Strategic Risk)

CLASSIFICATION GUIDELINES:
- "specific": When the text mentions a specific cause, hazard, or trigger (e.g., "from fire hazards", "from flood", "from labor dispute")
- "general": When the text describes a risk category without specifying the cause or trigger
- "too general": When the text represents the highest level risk category

CLASSIFICATION INSTRUCTIONS:
- Classify the granularity level based on the main TEXT TO CLASSIFY
- Use the ADDITIONAL CONTEXT only for better understanding and context, not for determining granularity level
- The additional context provides more specific details but should not influence the granularity classification

For "too general" classification, use the provided reference categories if available.""",
                "human": "Please classify the granularity level of the following RISK-related text:",
            },
            "control": ...,  # NOTE: we will add this later
            "rootcause": ...,  # NOTE: we will add this later
            "process": ...,  # NOTE: we will add this later
        }

        # Get the appropriate prompt for the context
        if context not in domain_prompts:
            raise ValueError(
                f"Unsupported context: {context}. Must be one of {list(domain_prompts.keys())}"
            )

        prompt_config = domain_prompts[context]

        # Create the prompt template
        system_prompt = prompt_config["system"]
        human_prompt = f"""{prompt_config["human"]}

TEXT TO CLASSIFY: {text}

ADDITIONAL CONTEXT (more specific details): {description}

{f"REFERENCE CATEGORIES: {', '.join(reference_categories)}" if reference_categories else ""}

IMPORTANT: Classify the granularity level based on the TEXT TO CLASSIFY. The ADDITIONAL CONTEXT provides more specific details to help with classification but should not be the basis for the granularity level.

Provide your classification in the specified format."""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

        # Create structured output
        structured_llm = llm.with_structured_output(GranularityClassification)
        chain = prompt | structured_llm

        # Invoke the chain
        result = chain.invoke({})

        return result.model_dump()

    except Exception as e:
        print(f"Error classifying granularity: {str(e)}")
        raise Exception(f"Error classifying granularity: {str(e)}")


def get_granularity_statistics(classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from granularity classifications.

    Args:
        classifications: List of classification results

    Returns:
        Dictionary containing statistics
    """

    if not classifications:
        return {}

    level_counts = {}
    confidence_scores = []
    valid_classifications = [c for c in classifications if c.get("level") is not None]

    for classification in valid_classifications:
        level = classification.get("level")
        confidence = classification.get("confidence", 0)

        if level:
            level_counts[level] = level_counts.get(level, 0) + 1
        if confidence:
            confidence_scores.append(confidence)

    return {
        "total_texts": len(classifications),
        "valid_classifications": len(valid_classifications),
        "level_distribution": level_counts,
        "average_confidence": (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        ),
        "most_common_level": (
            max(level_counts.items(), key=lambda x: x[1])[0] if level_counts else None
        ),
    }


RISK_CATEGORIES = [
    "Cybersecurity Risk",
    "Operational Risk",
    "Financial Risk",
    "Strategic Risk",
    "Reputational Risk",
]


# Example usage and testing
if __name__ == "__main__":
    # Example texts for each domain (ordered by granularity level: too specific, specific, general, too general)
    risk_texts = [
        "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM",  # too specific
        "Employee clicked on phishing email",  # specific
        "Email security breach",  # general
        "Cybersecurity Risk",  # too general
    ]

    control_texts = [
        "Firewall rule #12345 blocks port 443 for IP range 192.168.1.0/24",  # too specific
        "Firewall blocks unauthorized access",  # specific
        "Network security controls",  # general
        "Technical Controls",  # too general
    ]

    root_cause_texts = [
        "Database server crashed due to disk space exhaustion on /var/log partition at 3:45 AM",  # too specific
        "Database server crashed due to disk space issues",  # specific
        "System resource issues",  # general
        "System Failures",  # too general
    ]

    process_texts = [
        "User John Smith submitted expense report #12345 on March 15th using Chrome browser version 120.0.6099.109",  # too specific
        "User submitted expense report",  # specific
        "Document submission process",  # general
        "Financial Processes",  # too general
    ]

    # Example reference categories for each domain
    risk_categories = [
        "Cybersecurity Risk",
        "Operational Risk",
        "Human Error",
        "External Threats",
    ]
    control_categories = [
        "Technical Controls",
        "Administrative Controls",
        "Physical Controls",
        "Detective Controls",
    ]
    root_cause_categories = [
        "System Failures",
        "Resource Management",
        "Infrastructure Issues",
        "Human Factors",
    ]
    process_categories = [
        "Financial Processes",
        "HR Processes",
        "IT Processes",
        "Operational Processes",
    ]

    # Test different domain classifications
    print("=== TESTING DIFFERENT DOMAIN CLASSIFICATIONS ===")

    # Test RISK domain
    print("\n--- RISK Domain ---")
    for i, text in enumerate(risk_texts):
        result = classify_granularity(
            text=text,  # This is what gets classified for granularity
            context="risk",
            description="Risk assessment text for cybersecurity domain - this provides additional context but doesn't affect granularity level",
            reference_categories=risk_categories,
        )
        print(
            f"Risk Text {i+1}: {result['level']} (Confidence: {result['confidence']})"
        )
        print(f"  Text: {text}")
        print(f"  Reasoning: {result['reasoning']}")
        print()

    # Test CONTROL domain
    print("--- CONTROL Domain ---")
    for i, text in enumerate(control_texts):
        result = classify_granularity(
            text=text,
            context="control",
            description="Control and mitigation measures for IT security",
            reference_categories=control_categories,
        )
        print(
            f"Control Text {i+1}: {result['level']} (Confidence: {result['confidence']})"
        )
        print(f"  Text: {text}")
        print(f"  Reasoning: {result['reasoning']}")
        print()

    # Test ROOT CAUSE domain
    print("--- ROOT CAUSE Domain ---")
    for i, text in enumerate(root_cause_texts):
        result = classify_granularity(
            text=text,
            context="rootcause",
            description="Root cause analysis for system failures",
            reference_categories=root_cause_categories,
        )
        print(
            f"Root Cause Text {i+1}: {result['level']} (Confidence: {result['confidence']})"
        )
        print(f"  Text: {text}")
        print(f"  Reasoning: {result['reasoning']}")
        print()

    # Test PROCESS domain
    print("--- PROCESS Domain ---")
    for i, text in enumerate(process_texts):
        result = classify_granularity(
            text=text,
            context="process",
            description="Business process and workflow analysis",
            reference_categories=process_categories,
        )
        print(
            f"Process Text {i+1}: {result['level']} (Confidence: {result['confidence']})"
        )
        print(f"  Text: {text}")
        print(f"  Reasoning: {result['reasoning']}")
        print()

    # Test statistics with all results
    print("=== TESTING STATISTICS ===")
    all_results = []
    for text, domain, categories, desc in [
        (risk_texts[1], "risk", risk_categories, "Risk assessment text"),
        (control_texts[1], "control", control_categories, "Control measures"),
        (
            root_cause_texts[1],
            "rootcause",
            root_cause_categories,
            "Root cause analysis",
        ),
        (process_texts[1], "process", process_categories, "Business process"),
    ]:
        result = classify_granularity(
            text=text, context=domain, description=desc, reference_categories=categories
        )
        result["input_text"] = text
        result["domain"] = domain
        all_results.append(result)

    stats = get_granularity_statistics(all_results)
    print(f"Statistics: {stats}")
