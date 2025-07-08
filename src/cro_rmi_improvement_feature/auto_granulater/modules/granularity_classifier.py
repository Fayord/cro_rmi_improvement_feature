# The granularity_classifier is designed to programmatically assess and categorize input text into a multi-level depth model, providing classifications at varying degrees of specificity.

# Purpose: To classify a given phrase or sentence into one of three defined granularity levels:


# Level 1 (Too Specific): Captures atomic expressions or distinct factoids, providing maximum detail.

# Level 2 (general): Denotes a more generalized but still specific event or concept.

# Level 3 (Broad Category): Represents the overarching topic or domain, which must always be populated.

# Conditional Output: A key feature is its ability to output to be 1,2,3
# for Level 3 we have set of data to use as a reference

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
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Load environment variables
dir_path = os.path.dirname(os.path.realpath(__file__))
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

    level: Literal[0, 1, 2, 3] = Field(
        description="Granularity level: 0=Too Specific, 1=Specific, 2=General, 3=Broad Category"
    )

    confidence: int = Field(
        ge=1, le=5, description="Confidence level from 1 (low) to 5 (high)"
    )

    reasoning: str = Field(description="Brief explanation of why this level was chosen")


def get_llm(provider: str = "openai", model_name: Optional[str] = None) -> ChatOpenAI:
    """Initialize an LLM based on the provider."""
    if provider == "openai":
        model = model_name if model_name else "gpt-4o-mini"
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
        reference_categories: Optional list of reference categories for Level 3 classification
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
- Level 0 (Too Specific): Captures atomic expressions or distinct factoids with maximum detail
  Examples: "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM"
  
- Level 1 (Specific): Denotes specific events or concepts with moderate detail
  Examples: "Employee clicked on phishing email", "Phishing attack targeting employees"
  
- Level 2 (General): Denotes a more generalized but still specific event or concept
  Examples: "Email security breach", "Social engineering attack"
  
- Level 3 (Broad Category): Represents the overarching topic or domain
  Examples: "Cybersecurity Risk", "Human Error", "External Threats"

RISK DOMAIN CONTEXT:
- Focus on potential events, threats, vulnerabilities, or exposures that could cause harm
- Consider likelihood, impact, and severity of the risk
- Look for specific risk events, risk scenarios, or risk factors
- For Level 3, categorize into broad risk types (e.g., Operational Risk, Financial Risk, Strategic Risk)

CLASSIFICATION INSTRUCTIONS:
- Classify the granularity level based on the main TEXT TO CLASSIFY
- Use the ADDITIONAL CONTEXT only for better understanding and context, not for determining granularity level
- The additional context provides more specific details but should not influence the granularity classification

For Level 3 classification, use the provided reference categories if available.""",
                "human": "Please classify the granularity level of the following RISK-related text:",
            },
            "control": {
                "system": """You are an expert in risk assessment and business analysis. Your task is to classify the granularity level of input text specifically related to CONTROL and MITIGATION measures.

GRANULARITY LEVELS:
- Level 0 (Too Specific): Captures atomic expressions or distinct factoids with maximum detail
  Examples: "Firewall rule #12345 blocks port 443 for IP range 192.168.1.0/24"
  
- Level 1 (Specific): Denotes specific events or concepts with moderate detail
  Examples: "Firewall blocks unauthorized access", "Access control measures"
  
- Level 2 (General): Denotes a more generalized but still specific event or concept
  Examples: "Network security controls", "Access management"
  
- Level 3 (Broad Category): Represents the overarching topic or domain
  Examples: "Technical Controls", "Administrative Controls", "Physical Controls"

CONTROL/MITIGATION DOMAIN CONTEXT:
- Focus on measures, procedures, policies, or mechanisms to prevent, detect, or mitigate risks
- Consider preventive, detective, and corrective controls
- Look for specific control implementations, security measures, or mitigation strategies
- For Level 3, categorize into broad control types (e.g., Technical Controls, Administrative Controls, Physical Controls)

CLASSIFICATION INSTRUCTIONS:
- Classify the granularity level based on the main TEXT TO CLASSIFY
- Use the ADDITIONAL CONTEXT only for better understanding and context, not for determining granularity level
- The additional context provides more specific details but should not influence the granularity classification

For Level 3 classification, use the provided reference categories if available.""",
                "human": "Please classify the granularity level of the following CONTROL/MITIGATION-related text:",
            },
            "rootcause": {
                "system": """You are an expert in risk assessment and business analysis. Your task is to classify the granularity level of input text specifically related to ROOT CAUSE analysis.

GRANULARITY LEVELS:
- Level 0 (Too Specific): Captures atomic expressions or distinct factoids with maximum detail
  Examples: "Database server crashed due to disk space exhaustion on /var/log partition at 3:45 AM"
  
- Level 1 (Specific): Denotes specific events or concepts with moderate detail
  Examples: "Database server crashed due to disk space issues", "System failure due to resource exhaustion"
  
- Level 2 (General): Denotes a more generalized but still specific event or concept
  Examples: "System resource issues", "Infrastructure problems"
  
- Level 3 (Broad Category): Represents the overarching topic or domain
  Examples: "System Failures", "Resource Management", "Infrastructure Issues"

ROOT CAUSE DOMAIN CONTEXT:
- Focus on underlying factors, fundamental issues, or primary causes that lead to risks or incidents
- Consider systemic issues, organizational factors, or technical problems
- Look for specific causal factors, contributing conditions, or fundamental problems
- For Level 3, categorize into broad root cause types (e.g., Human Factors, Technical Issues, Process Failures)

CLASSIFICATION INSTRUCTIONS:
- Classify the granularity level based on the main TEXT TO CLASSIFY
- Use the ADDITIONAL CONTEXT only for better understanding and context, not for determining granularity level
- The additional context provides more specific details but should not influence the granularity classification

For Level 3 classification, use the provided reference categories if available.""",
                "human": "Please classify the granularity level of the following ROOT CAUSE-related text:",
            },
            "process": {
                "system": """You are an expert in risk assessment and business analysis. Your task is to classify the granularity level of input text specifically related to BUSINESS PROCESSES and procedures.

GRANULARITY LEVELS:
- Level 0 (Too Specific): Captures atomic expressions or distinct factoids with maximum detail
  Examples: "User John Smith submitted expense report #12345 on March 15th using Chrome browser version 120.0.6099.109"
  
- Level 1 (Specific): Denotes specific events or concepts with moderate detail
  Examples: "User submitted expense report", "Expense approval process"
  
- Level 2 (General): Denotes a more generalized but still specific event or concept
  Examples: "Document submission process", "Approval workflows"
  
- Level 3 (Broad Category): Represents the overarching topic or domain
  Examples: "Financial Processes", "HR Processes", "IT Processes"

PROCESS DOMAIN CONTEXT:
- Focus on business processes, workflows, procedures, or operational activities
- Consider process steps, workflows, procedures, or operational activities
- Look for specific process activities, workflow steps, or procedural elements
- For Level 3, categorize into broad process types (e.g., Financial Processes, HR Processes, IT Processes)

CLASSIFICATION INSTRUCTIONS:
- Classify the granularity level based on the main TEXT TO CLASSIFY
- Use the ADDITIONAL CONTEXT only for better understanding and context, not for determining granularity level
- The additional context provides more specific details but should not influence the granularity classification

For Level 3 classification, use the provided reference categories if available.""",
                "human": "Please classify the granularity level of the following PROCESS-related text:",
            },
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
    # Example texts for each domain (ordered by granularity level: 0, 1, 2, 3)
    risk_texts = [
        "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM",  # Level 0
        "Employee clicked on phishing email",  # Level 1
        "Email security breach",  # Level 2
        "Cybersecurity Risk",  # Level 3
    ]

    control_texts = [
        "Firewall rule #12345 blocks port 443 for IP range 192.168.1.0/24",  # Level 0
        "Firewall blocks unauthorized access",  # Level 1
        "Network security controls",  # Level 2
        "Technical Controls",  # Level 3
    ]

    root_cause_texts = [
        "Database server crashed due to disk space exhaustion on /var/log partition at 3:45 AM",  # Level 0
        "Database server crashed due to disk space issues",  # Level 1
        "System resource issues",  # Level 2
        "System Failures",  # Level 3
    ]

    process_texts = [
        "User John Smith submitted expense report #12345 on March 15th using Chrome browser version 120.0.6099.109",  # Level 0
        "User submitted expense report",  # Level 1
        "Document submission process",  # Level 2
        "Financial Processes",  # Level 3
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
            f"Risk Text {i+1}: Level {result['level']} (Confidence: {result['confidence']})"
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
            f"Control Text {i+1}: Level {result['level']} (Confidence: {result['confidence']})"
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
            f"Root Cause Text {i+1}: Level {result['level']} (Confidence: {result['confidence']})"
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
            f"Process Text {i+1}: Level {result['level']} (Confidence: {result['confidence']})"
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
