import random
from typing import Any, Dict, List, Optional, Literal
import os
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp

import pickle

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np
from collections import Counter

# import re
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    env_path = f"{dir_path}/../confidential/.env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "API key is missing"
except Exception:
    # /Users/ford/Documents/coding_trae/cro_rmi_improvement_feature/src/cro_rmi_improvement_feature/plot_risk_related/utils.py
    print(dir_path)
    env_path = f"{dir_path}/../../../../../coding/confidential/.env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "API key is missing"


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


def classify_final_edge_relationship(
    edge_relationship_a_b: Dict[str, str],
    edge_relationship_b_a: Dict[str, str],
    risk_a_name: str,
    risk_b_name: str,
) -> dict:
    # edge_relationship_a_b, edge_relationship_b_a, risk_a["risk"], risk_b["risk"]
    try:
        llm = get_llm()
        # parser = PydanticOutputParser(pydantic_object=EdgeRelationship)

        # Format input data for the prompt
        edge_relationship_a_b_reason = edge_relationship_a_b["reason"]
        edge_relationship_b_a_reason = edge_relationship_b_a["reason"]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert in risk assessment and business analysis. Your task is to classify the relationship between two risks based on their descriptions and context. your analysis will be from the perspective of business owner.

Analyze the provided data for {risk_a_name} and {risk_b_name} and determine the relationship type.

Provide the classification and a brief reason in the specified JSON format.

""",
                ),
                (
                    "human",
                    f"Please classify the relationship between risk{risk_a_name} and {risk_b_name} with these two reason: 1.{edge_relationship_a_b_reason} 2. {edge_relationship_b_a_reason}",
                ),
            ]
        )
        json_schema = {
            "properties": {
                "relationship": {
                    "description": f"Relationship between two risks. {risk_a_name}_caused_by_{risk_b_name} means {risk_b_name} is the cause and {risk_a_name} is the effect. {risk_b_name}_caused_by_{risk_a_name} means {risk_a_name} is the cause and {risk_b_name} is the effect.  no_relationship means there is no relationship between two risks. be_a_cause_to_each_other means two risks are both cause and effect to each other.",
                    "enum": [
                        f"{risk_a_name}_caused_by_{risk_b_name}",
                        f"{risk_b_name}_caused_by_{risk_a_name}",
                        "no_relationship",
                        "be_a_caused_by_each_other",
                    ],
                    "title": "Relationship",
                    "type": "string",
                },
                "reason": {
                    "description": "Brief reason for the relationship classification in 1 sentence",
                    "title": "Reason",
                    "type": "string",
                },
            },
            "required": ["relationship", "reason"],
            "title": "EdgeRelationship",
            "type": "object",
        }
        structured_llm = llm.with_structured_output(json_schema)
        chain = prompt | structured_llm

        # Invoke the chain
        result = chain.invoke({})
        # print(f"{risk_a_str=}")
        # Return the relationship field from the parsed model
        return result

    except Exception as e:
        print(f"Error classifying edge relationship: {str(e)}")
        raise Exception(f"Error classifying edge relationship: {str(e)}")


def classify_edge_relationship(risk_a: Dict[str, Any], risk_b: Dict[str, Any]) -> dict:
    try:
        llm = get_llm()
        # parser = PydanticOutputParser(pydantic_object=EdgeRelationship)

        # Format input data for the prompt
        risk_a_str = "\n".join([f"- {k}: {v}" for k, v in risk_a.items()])
        risk_b_str = "\n".join([f"- {k}: {v}" for k, v in risk_b.items()])
        risk_a_name = risk_a["risk"]
        risk_b_name = risk_b["risk"]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert in risk assessment and business analysis. Your task is to classify the relationship between two risks based on their descriptions and context. your analysis will be from the perspective of business owner.

Analyze the provided data for {risk_a_name} and {risk_b_name} and determine the relationship type.


Provide the classification and a brief reason in the specified JSON format.

""",
                ),
                (
                    "human",
                    f"Please classify the relationship between {risk_a_str} and  {risk_b_str}.",
                ),
            ]
        )
        json_schema = {
            "properties": {
                "relationship": {
                    "description": f"Relationship between two risks. {risk_a_name}_caused_by_{risk_b_name} means {risk_b_name} is the cause and {risk_a_name} is the effect. {risk_b_name}_caused_by_{risk_a_name} means {risk_a_name} is the cause and {risk_b_name} is the effect.  no_relationship means there is no relationship between two risks. be_a_cause_to_each_other means two risks are both cause and effect to each other.",
                    "enum": [
                        f"{risk_a_name}_caused_by_{risk_b_name}",
                        f"{risk_b_name}_caused_by_{risk_a_name}",
                        "no_relationship",
                        "be_a_caused_by_each_other",
                    ],
                    "title": "Relationship",
                    "type": "string",
                },
                "reason": {
                    "description": "Brief reason for the relationship classification in 1 sentence",
                    "title": "Reason",
                    "type": "string",
                },
            },
            "required": ["relationship", "reason"],
            "title": "EdgeRelationship",
            "type": "object",
        }
        structured_llm = llm.with_structured_output(json_schema)
        chain = prompt | structured_llm

        # Invoke the chain
        result = chain.invoke({})
        # print(f"{risk_a_str=}")
        # Return the relationship field from the parsed model
        return result

    except Exception as e:
        print(f"Error classifying edge relationship: {str(e)}")
        raise Exception(f"Error classifying edge relationship: {str(e)}")


def tell_a_story_risk_data(
    input_risk_data: Dict[str, Any],
    related_risk_data: List[Dict[str, Any]],
    additional_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a concise plain text story about cause and effect based on risk data.

    Args:
        input_risk_data: Dictionary containing the main risk data.
        related_risk_data: List of dictionaries containing related risk data.
        additional_data: Optional dictionary containing additional business context.

    Returns:
        A concise plain text story about cause and effect.
    """
    try:
        llm = get_llm()
        # Removed JsonOutputParser as output is plain text

        # Format input data for the prompt
        input_data_str = "\n".join([f"- {k}: {v}" for k, v in input_risk_data.items()])

        related_data_str = ""
        if related_risk_data:
            related_data_str = "\nRelated Risk Data:\n"
            for i, item in enumerate(related_risk_data):
                related_data_str += f"Item {i+1}:\n"
                related_data_str += (
                    "\n".join([f"  - {k}: {v}" for k, v in item.items()]) + "\n"
                )

        additional_data_str = ""
        if additional_data:
            additional_data_str = "\nAdditional Business Context:\n"
            additional_data_str += "\n".join(
                [f"- {k}: {v}" for k, v in additional_data.items()]
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert in risk assessment and business analysis. Your task is to create a concise plain text story (maximum 3-5 sentences) that describes the cause and effect relationship based on the provided risk data and context. Focus on the business aspect of the cause and effect.

Input Risk Data:
{input_data_str}

{related_data_str}
{additional_data_str}

Generate the story in plain text, focusing on clarity and conciseness.
""",
                ),
                (
                    "human",
                    "Please generate the cause and effect story based on the data above.",
                ),
            ]
        )

        chain = prompt | llm  # Removed parser

        # Invoke the chain
        result = chain.invoke(
            {}
        )  # No specific input variable needed for invoke with this prompt structure

        return result.content.strip()  # Return plain text content
    except Exception as e:
        print(f"Error generating improved example/story: {str(e)}")
        return f"Error generating story: {str(e)}"


def tell_a_story_risk_data_grouped(
    input_risk_data: Dict[str, Any],
    related_risk_data: List[Dict[str, Any]],
    additional_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a concise plain text story about cause and effect, grouping related risks.

    Args:
        input_risk_data: Dictionary containing the main risk data.
        related_risk_data: List of dictionaries containing related risk data.
        additional_data: Optional dictionary containing additional business context.

    Returns:
        A concise plain text story about cause and effect, with related risks grouped.
    """
    try:
        llm = get_llm()

        # Format input data for the prompt
        input_data_str = "\n".join([f"- {k}: {v}" for k, v in input_risk_data.items()])

        related_data_str = ""
        if related_risk_data:
            related_data_str = "\nRelated Risk Data:\n"
            for i, item in enumerate(related_risk_data):
                related_data_str += f"Item {i+1}:\n"
                related_data_str += (
                    "\n".join([f"  - {k}: {v}" for k, v in item.items()]) + "\n"
                )

        additional_data_str = ""
        if additional_data:
            additional_data_str = "\nAdditional Business Context:\n"
            additional_data_str += "\n".join(
                [f"- {k}: {v}" for k, v in additional_data.items()]
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert in risk assessment and business analysis. Your task is to create a concise plain text story (maximum 3-5 sentences) that describes the cause and effect relationship based on the provided risk data and context. Focus on the business aspect of the cause and effect.

When analyzing the 'Related Risk Data', identify common themes or connections between the items and group them logically in your analysis to provide a more streamlined and concise story. Do not list out each related risk individually in the final story, but rather synthesize their collective impact or relationship to the main risk.

Input Risk Data:
{input_data_str}

{related_data_str}
{additional_data_str}

Generate the story in plain text, focusing on clarity and conciseness, and reflecting the grouped analysis of related risks.
""",
                ),
                (
                    "human",
                    "Please generate the cause and effect story based on the data above.",
                ),
            ]
        )

        chain = prompt | llm

        # Invoke the chain
        result = chain.invoke({})

        return result.content.strip()
    except Exception as e:
        print(f"Error generating grouped story: {str(e)}")
        return f"Error generating grouped story: {str(e)}"


def find_equal_count_boundaries(numbers: list[float], n_sections: int):
    """
    Given a list of numbers, find the boundary values that split the list into n_sections
    such that each section has (as close as possible) the same number of elements.
    Returns a list of boundary values (length n_sections+1).
    """
    assert n_sections > 0, "n_sections must be greater than 0"
    if not numbers or n_sections < 1:
        return []

    sorted_numbers = sorted(numbers)
    total = len(sorted_numbers)
    boundaries = [sorted_numbers[0]]
    for i in range(1, n_sections):
        idx = int(round(i * total / n_sections))
        # Clamp index to valid range
        idx = min(idx, total - 1)
        boundaries.append(sorted_numbers[idx])
    boundaries.append(sorted_numbers[-1])
    # -1 for min and +1 for max
    # boundaries[0] -= 1
    boundaries[-1] += 1
    return boundaries


def find_proportional_count_boundaries(
    numbers: list[float], proportion_member_list: list[int]
):
    """
    Given a list of numbers and a list of proportions (as percentages summing to 100),
    find the boundary values that split the list into sections according to those proportions.
    Returns a list of boundary values (length len(proportion_member_list)+1).
    """
    assert sum(proportion_member_list) == 100, "Sum of proportions must be 100"
    if not numbers or not proportion_member_list:
        return []

    sorted_numbers = sorted(numbers)
    total = len(sorted_numbers)
    boundaries = [sorted_numbers[0]]
    cumulative = 0
    for proportion in proportion_member_list[:-1]:
        cumulative += proportion
        idx = int(round(cumulative * total / 100))
        idx = min(idx, total - 1)
        boundaries.append(sorted_numbers[idx])
    boundaries.append(sorted_numbers[-1])
    boundaries[-1] += 1
    return boundaries


def get_level_from_boundaries(boundaries: list[int], test_number: float):
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if start <= test_number < end:
            return i + 1
    raise ValueError(f"Number {test_number} is outside the range of boundaries.")


if __name__ == "__main__":

    numbers = random.sample(range(1, 201), 100)  # 100 unique numbers from 1 to 100
    numbers.sort()
    print(numbers)
    print("numbers", len(numbers))
    boundaries = find_proportional_count_boundaries(numbers, [40, 40, 10, 10])

    # display member of each section
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        section = [num for num in numbers if start <= num < end]

        print(f"Section  {i + 1}: {len(section)} {section}")
    test_number = 170
    level = get_level_from_boundaries(boundaries, test_number)
    print(level)
    print(boundaries)
