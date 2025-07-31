# risk_interdependency_llm.py

import openai
from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks import get_openai_callback
import random
import json
from dotenv import load_dotenv
import os
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import pandas as pd
from typing import Dict, Any

dir_path = os.path.dirname(os.path.abspath(__file__))

set_llm_cache(SQLiteCache(database_path=f"{dir_path}/.relationship_classifier.db"))

env_path = os.path.join(dir_path, "../../../../.env")
load_dotenv(env_path)
# print abs env path
print(f"env_path: {os.path.abspath(env_path)}")
# if OPENAI_API_KEY not set, raise error
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")


class RiskRelationResult(BaseModel):
    interdependency_type: Literal[
        "Causal",
        "Correlated",
        "None",
    ]
    direction: Optional[
        Literal[
            "Both",
            "A → B",
            "B → A",
            "None",
        ]
    ] = None
    rationale: str
    confidence: int = Field(..., ge=1, le=5)


class RiskSummary(BaseModel):
    risk_desc: str
    rootcause: str
    process: str


# === LangChain Setup ===


def get_llm(model_name="o3-mini"):
    if model_name == "gpt-4.1-mini":
        return ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.1,
        )
    elif model_name == "gpt-4o-mini":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
        )
    elif model_name == "o3-mini":
        return ChatOpenAI(
            model="o3-mini",
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


# === Call LangChain and Parse ===


def classify_relationship(
    risk_a: Dict[str, Any], risk_b: Dict[str, Any], analyze_model_name: str
) -> Dict[str, Any]:
    try:
        llm = get_llm(model_name=analyze_model_name)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a risk analysis assistant reviewing pairs of risk statements.
Analyze the interdependency between two risks and determine:
1. Interdependency Type
2. Direction (if directional)
3. Rationale (1-2 sentences)


Valid Types:

1. **Causal**: One risk directly causes or triggers the other. Can be unidirectional (A→B or B→A) or bidirectional (Both).
2. **Correlated**: The risks often occur together, but with no clear causal link or order.
3. **None**: There is no meaningful relationship.

Valid directions:
- Both (Bidirectional causality)
- A → B (Risk A leads to Risk B)
- B → A (Risk B leads to Risk A)
- None (For non-directional types)

Direction is required for Causal types.
For other types (Correlated, None), direction should be "None".

Provide the analysis in the specified JSON format.""",
                ),
                (
                    "human",
                    f"""Analyze the interdependency between the following two risks:
Risk A: {risk_a['risk']} \n Risk Description: {risk_a['risk_desc_summary']} \n Root Cause: {risk_a['rootcause_summary']} \n Process: {risk_a['process_summary']}
Risk B: {risk_b['risk']} \n Risk Description: {risk_b['risk_desc_summary']} \n Root Cause: {risk_b['rootcause_summary']} \n Process: {risk_b['process_summary']}""",
                ),
            ]
        )

        structured_llm = llm.with_structured_output(RiskRelationResult)
        chain = prompt | structured_llm

        # Invoke the chain
        result = chain.invoke({})

        # Convert to RiskRelationResult model for validation
        validated_result = result.model_dump()
        direction = validated_result["direction"]
        interdependency_type = validated_result["interdependency_type"]
        if interdependency_type in ["Causal"]:
            if direction == "None":
                raise ValueError(f"direction is None for {interdependency_type}")
        else:
            if direction != "None":
                raise ValueError(f"direction is not None for {interdependency_type}")
        return validated_result

    except Exception as e:
        print(f"[ERROR] Failed to analyze pair {str(e)}")
        raise e


def final_relationship(
    interdependency_type_a_b,
    interdependency_type_b_a,
    direction_a_b,
    direction_b_a,
):
    priority_interdependency_type_list = [
        "Causal",
        "Correlated",
        "None",
    ]
    priority_direction_list = [
        "Both",
        "A → B",
        "B → A",
        "None",
    ]

    ####### preprocess
    if interdependency_type_b_a in ["Causal"]:
        if direction_b_a == "A → B":
            direction_b_a = "B → A"
        elif direction_b_a == "B → A":
            direction_b_a = "A → B"
    #######
    if interdependency_type_a_b == interdependency_type_b_a:
        final_interdependency_type = interdependency_type_a_b
        if interdependency_type_a_b in ["Causal"]:

            index_direction_a_b = priority_direction_list.index(direction_a_b)
            index_direction_b_a = priority_direction_list.index(direction_b_a)
            if (index_direction_a_b, index_direction_b_a) in [
                ("A → B", "B → A"),
                ("B → A", "A → B"),
            ]:
                final_direction = "Both"
            elif index_direction_a_b < index_direction_b_a:
                final_direction = direction_a_b
            else:
                final_direction = direction_b_a
        else:
            final_direction = "None"

    else:
        index_interdependency_type_a_b = priority_interdependency_type_list.index(
            interdependency_type_a_b
        )
        index_interdependency_type_b_a = priority_interdependency_type_list.index(
            interdependency_type_b_a
        )
        if index_interdependency_type_a_b < index_interdependency_type_b_a:
            final_interdependency_type = interdependency_type_a_b
            final_direction = direction_a_b
        else:
            final_interdependency_type = interdependency_type_b_a
            final_direction = direction_b_a

    return final_interdependency_type, final_direction


# === Main Loop ===

if __name__ == "__main__":
    random.seed(42)

    dir_path = os.path.dirname(os.path.abspath(__file__))
