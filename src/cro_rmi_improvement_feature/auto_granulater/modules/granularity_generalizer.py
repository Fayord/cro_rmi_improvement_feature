# The granularity_generalizer takes an input text and generates a more generalized version of it, effectively moving the text "up" the granularity hierarchy.

# Purpose: To transform a given text into a more abstract or broader representation, reducing its level of detail while retaining core meaning. This is particularly useful for tasks like summarization or creating higher-level overviews.

# Granularity Control: This function allows for controlled text generation, enabling the system to produce summaries or expanded texts at specific levels of detail.

# the level of granularity is 4 levels: too specific, specific, general, too general
# please make sure the level of Granularity is the same as granularity_classifier.py mentions

import os
from typing import Dict, Any, Optional, Literal, List
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


class GranularityVersions(BaseModel):
    """Structured output for granularity versions."""

    too_specific: Optional[str] = Field(
        default=None, description="Version at 'too specific' level (most detailed)"
    )

    specific: Optional[str] = Field(
        default=None, description="Version at 'specific' level (moderate detail)"
    )

    general: Optional[str] = Field(
        default=None, description="Version at 'general' level (less detail)"
    )

    too_general: Optional[str] = Field(
        default=None, description="Version at 'too general' level (least detail)"
    )


def get_llm(provider: str = "openai", model_name: Optional[str] = None) -> ChatOpenAI:
    """Initialize an LLM based on the provider."""
    if provider == "openai":
        model = model_name if model_name else "gpt-4.1-mini"
        temperature = 0.1 if model_name not in ["o3-mini"] else None
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_granularity_hierarchy() -> Dict[str, int]:
    """Get the granularity hierarchy mapping levels to their order (0 = most specific, 3 = most general)."""
    return {"too specific": 0, "specific": 1, "general": 2, "too general": 3}


def granularity_generalizer(
    text: str,
    description: str,
    granularity: Literal["too specific", "specific", "general", "too general"],
    context: Literal["risk", "control", "rootcause", "process"],
    reference_categories: Optional[List[str]] = None,
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Generate versions of the input text at different granularity levels.

    Logic:
    - For the same level as input: return original text
    - For more specific levels: return None (cannot make more specific)
    - For more general levels: use LLM to generate generalized version

    Args:
        text: The input text to generalize
        description: Additional context/description for the text
        granularity: The current granularity level of the input text
        context: The domain context ("risk", "control", "rootcause", "process")
        reference_categories: Optional list of reference categories for "too general" classification
        model_name: The LLM model to use for generation

    Returns:
        Dictionary containing versions at each granularity level
    """

    try:
        # Get granularity hierarchy
        hierarchy = get_granularity_hierarchy()
        input_level = hierarchy[granularity]

        # Initialize result with None for all levels
        result = {
            "too_specific": None,
            "specific": None,
            "general": None,
            "too_general": None,
        }

        # Set the original text for the input level
        result[granularity.replace(" ", "_")] = text

        # Determine which levels need to be generated (more general than input)
        levels_to_generate = []
        for level, order in hierarchy.items():
            if order > input_level:  # More general than input
                levels_to_generate.append(level)

        if not levels_to_generate:
            return result

        # Generate more general versions using LLM
        llm = get_llm(model_name=model_name)

        # Define domain-specific prompts
        domain_prompts = {
            "risk": {
                "system": """You are an expert in risk assessment and business analysis. Your task is to create more general versions of input text specifically related to RISK identification and assessment.

GRANULARITY LEVELS (from most specific to most general):
- "too specific": Maximum detail with atomic expressions or distinct factoids
  Examples: "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM"
- "specific": Moderate detail with clear cause/type but less specific details
  Examples: "Employee clicked on phishing email", "Business interruption from fire hazards"
- "general": Broader categories without specific causes or details - should be ABSTRACT and HIGH-LEVEL
  Examples: "Email security breach", "Business interruption", "Cybersecurity breach", "Operational disruption"
- "too general": Overarching topic or domain level
  Examples: "Cybersecurity Risk", "Operational Risk", "Financial Risk"

RISK DOMAIN CONTEXT:
- Focus on potential events, threats, vulnerabilities, or exposures that could cause harm
- Consider likelihood, impact, and severity of the risk
- For "too general" level, categorize into broad risk types

GENERALIZATION GUIDELINES:
- Remove specific details, names, dates, times, exact locations
- Replace specific instances with broader categories
- Maintain the core meaning and context
- For "general" level: REMOVE ALL CAUSE-SPECIFIC DETAILS and focus on the BROAD RISK CATEGORY
- For "general" level: Do NOT include "due to", "from", or cause-specific phrases
- For "general" level: Use simple, abstract risk categories like "Business interruption", "Data breach", "System failure"
- For "too general" level, use provided reference categories if available

EXAMPLES OF GOOD GENERALIZATION:
Input: "Business interruption from natural disasters"
- General: "Business interruption" (NOT "Business interruption due to natural disasters")
- Too General: "Operational Risk"

Input: "Business interruption from labor dispute"
- General: "Business interruption" (NOT "Business interruption due to labor issues")
- Too General: "Operational Risk"

Input: "Business interruption from pandemic or epidemic"
- General: "Business interruption" (NOT "Business interruption due to health crises")
- Too General: "Operational Risk"

Input: "Data breach from unauthorized access"
- General: "Data breach" (NOT "Data breach due to unauthorized access")
- Too General: "Cybersecurity Risk"

Input: "System failure from hardware malfunction"
- General: "System failure" (NOT "System failure due to hardware issues")
- Too General: "Operational Risk"

CRITICAL: For "general" level, focus on the BROAD RISK TYPE without any cause-specific details.""",
                "human": "Please generate more general versions of the following RISK-related text:",
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

        human_prompt = f"""{prompt_config["human"]}

ORIGINAL TEXT: {text}
CONTEXT/DESCRIPTION: {description}
CURRENT GRANULARITY LEVEL: {granularity}
DOMAIN CONTEXT: {context}

{f"REFERENCE CATEGORIES: {', '.join(reference_categories)}" if reference_categories else ""}

Generate versions for these more general levels: {', '.join(levels_to_generate)}

{f"For 'too general' level, use the provided reference categories if available." if reference_categories else ""}

Provide your response in the specified format."""

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_config["system"]), ("human", human_prompt)]
        )

        # Create structured output
        structured_llm = llm.with_structured_output(GranularityVersions)
        chain = prompt | structured_llm

        # Invoke the chain
        generated_versions = chain.invoke({})

        # Update result with generated versions
        for level in levels_to_generate:
            level_key = level.replace(" ", "_")
            if hasattr(generated_versions, level_key):
                result[level_key] = getattr(generated_versions, level_key)

        return result

    except Exception as e:
        print(f"Error in granularity generalization: {str(e)}")
        raise Exception(f"Error in granularity generalization: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "text": "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM",
            "description": "Cybersecurity incident involving phishing attack",
            "granularity": "too specific",
            "context": "risk",
            "reference_categories": [
                "Cybersecurity Risk",
                "Operational Risk",
                "Financial Risk",
                "Strategic Risk",
            ],
            "expected_levels": ["specific", "general", "too general"],
        },
        {
            "text": "Firewall blocks unauthorized access",
            "description": "Network security control measure",
            "granularity": "specific",
            "context": "control",
            "reference_categories": [
                "Technical Controls",
                "Administrative Controls",
                "Physical Controls",
                "Detective Controls",
            ],
            "expected_levels": ["general", "too general"],
        },
        {
            "text": "Database server crashed due to disk space exhaustion",
            "description": "System failure root cause analysis",
            "granularity": "specific",
            "context": "rootcause",
            "reference_categories": [
                "System Failures",
                "Resource Management",
                "Infrastructure Issues",
                "Human Factors",
            ],
            "expected_levels": ["general", "too general"],
        },
        {
            "text": "User submitted expense report through online portal",
            "description": "Financial process for expense reporting",
            "granularity": "specific",
            "context": "process",
            "reference_categories": [
                "Financial Processes",
                "HR Processes",
                "IT Processes",
                "Operational Processes",
            ],
            "expected_levels": ["general", "too general"],
        },
        {
            "text": "Cybersecurity Risk",
            "description": "Broad risk category",
            "granularity": "too general",
            "context": "risk",
            "reference_categories": [
                "Cybersecurity Risk",
                "Operational Risk",
                "Financial Risk",
            ],
            "expected_levels": [],
        },
    ]

    print("=== TESTING GRANULARITY GENERALIZER ===")

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input Text: {test_case['text']}")
        print(f"Context: {test_case['context']}")
        print(f"Current Granularity: {test_case['granularity']}")
        print(f"Expected to generate: {test_case['expected_levels']}")

        try:
            result = granularity_generalizer(
                text=test_case["text"],
                description=test_case["description"],
                granularity=test_case["granularity"],
                context=test_case["context"],
                reference_categories=test_case["reference_categories"],
            )

            print("\nResults:")
            for level, version in result.items():
                status = "✓ Generated" if version else "✗ None (as expected)"
                print(f"  {level}: {status}")
                if version:
                    print(f"    Content: {version}")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 50)
