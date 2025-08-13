#!/usr/bin/env python3
"""
Data summarization module for the Risk Network Visualization System.

This module loads processed data and creates summaries using LLM.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import pandas as pd
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

dir_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(dir_path, "../../../../.env"))
USE_CACHED_LLM = os.getenv("USE_CACHED_LLM", True)
print(f"USE_CACHED_LLM: {USE_CACHED_LLM}")
USE_CACHED_LLM = os.getenv("USE_CACHED_LLM")
if USE_CACHED_LLM is None:
    USE_CACHED_LLM = True
elif USE_CACHED_LLM == "False":
    USE_CACHED_LLM = False
elif USE_CACHED_LLM == "True":
    USE_CACHED_LLM = True
else:
    raise ValueError(f"Unknown USE_CACHED_LLM: {USE_CACHED_LLM}")


# === LangChain Setup ===

from masked_data import create_mask_dict_from_excel, mask_data  # type: ignore


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


class RiskSummary(BaseModel):
    risk_desc_summary: Optional[str] = None
    rootcause_summary: Optional[str] = None
    process_summary: Optional[str] = None


def summarize_risk(risk_data: Dict[str, Any], is_masked: bool = True) -> Dict[str, Any]:
    mask_dict = create_mask_dict_from_excel()
    if is_masked:
        risk_data["risk"] = mask_data(risk_data["risk"], mask_dict)
        risk_data["risk_desc"] = mask_data(risk_data["risk_desc"], mask_dict)
        risk_data["rootcause_data"] = mask_data(risk_data["rootcause_data"], mask_dict)
        risk_data["process_data"] = mask_data(risk_data["process_data"], mask_dict)

    llm = get_llm(model_name="gpt-4.1-mini")
    risk_name = risk_data["risk"]
    risk_desc = risk_data["risk_desc"]
    assert risk_name != "", "Risk name is empty"
    assert risk_desc != "", "Risk description is empty"
    rootcause = risk_data["rootcause_data"]
    process = risk_data["process_data"]
    # Build the human_message dynamically, omitting empty fields as specified
    human_message_parts = [
        f"Risk: {risk_name}",
        f"Risk Description: {risk_desc}",
    ]
    if rootcause != "":
        human_message_parts.append(f"Root Cause: {rootcause}")
    if process != "":
        human_message_parts.append(f"Process: {process}")
    human_message_parts.append("Output in English.")
    human_message = "\n".join(human_message_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a risk analysis assistant. The input data may be in various languages, poorly ordered, or contain duplicates. Summarize the information in clear, concise English, preserving the original context and meaning as much as possible.",
            ),
            (
                "human",
                human_message,
            ),
        ]
    )
    structured_llm = llm.with_structured_output(RiskSummary)
    chain = prompt | structured_llm
    result = chain.invoke(risk_data)
    result = result.model_dump()
    return result


def load_processed_data(data_path: str) -> pd.DataFrame:
    """
    Load processed data from JSON format.

    Args:
        data_path: Path to the JSON file

    Returns:
        DataFrame with processed data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded processed data: {df.shape}")
    return df


def dummy_llm_summarize(text: str, summary_type: str = "general") -> str:
    """
    Dummy LLM summarize function - replace with your actual LLM function.

    Args:
        text: Text to summarize
        summary_type: Type of summary to generate

    Returns:
        Generated summary text
    """
    # TODO: Replace this with your actual LLM summarize function
    # This is a placeholder that creates a simple summary

    if not text or text.strip() == "":
        return "No content to summarize"

    # Simple dummy summarization logic
    words = text.split()
    if len(words) <= 10:
        return text  # Return original if too short

    # Take first few words as summary
    summary_words = words[: min(20, len(words))]
    summary = " ".join(summary_words)

    if len(words) > 20:
        summary += "..."

    return f"[{summary_type.upper()}] {summary}"


def create_risk_summary(row: pd.Series) -> Dict[str, str]:
    """
    Create a summary for a risk record using LLM.

    Args:
        row: DataFrame row containing risk data

    Returns:
        Dictionary with three summary fields
    """
    try:
        # Convert row to dict for LLM processing
        risk_data = {
            "risk": row.get("risk", ""),
            "risk_desc": row.get("risk_desc", ""),
            "rootcause_data": row.get("rootcause_data", ""),
            "process_data": row.get("process_data", ""),
        }

        # Use actual LLM to create summaries
        summaries = summarize_risk(risk_data)

        return {
            "risk_desc_summary": summaries.get("risk_desc_summary", ""),
            "rootcause_summary": summaries.get("rootcause_summary", ""),
            "process_summary": summaries.get("process_summary", ""),
        }

    except Exception as e:
        print(f"Error creating risk summary: {e}")
        # Return empty summaries on error
        return {"risk_desc_summary": "", "rootcause_summary": "", "process_summary": ""}


def add_summaries_to_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary fields to the DataFrame.

    Args:
        df: DataFrame with risk data

    Returns:
        DataFrame with added summary fields
    """
    print("Adding summaries to data...")

    # Add risk summaries using LLM
    print("Creating risk summaries using LLM...")
    risk_summaries = df.apply(create_risk_summary, axis=1)

    # Extract the three summary fields from the results
    df["risk_desc_summary"] = [
        summary.get("risk_desc_summary", "") for summary in risk_summaries
    ]
    df["rootcause_summary"] = [
        summary.get("rootcause_summary", "") for summary in risk_summaries
    ]
    df["process_summary"] = [
        summary.get("process_summary", "") for summary in risk_summaries
    ]

    print(f"Added summaries to {len(df)} records")
    print("Added summary fields: risk_desc_summary, rootcause_summary, process_summary")
    return df


def save_summarized_data(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame with summaries to JSON format.

    Args:
        df: DataFrame with summaries
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert DataFrame to list of dicts for JSON serialization
    data_to_save = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    print(f"Summarized data saved to: {output_path}")


def process_summaries(input_data_path: str, output_data_path: str) -> pd.DataFrame:
    """
    Process data and add summaries.

    Args:
        input_data_path: Path to the input data JSON file
        output_data_path: Path to save the summarized data JSON file

    Returns:
        DataFrame with added summaries
    """
    print("Starting data summarization process...")

    # Load processed data
    print("Loading processed data...")
    df = load_processed_data(input_data_path)

    # Add summaries
    print("Adding summaries...")
    df_with_summaries = add_summaries_to_data(df)

    # Save summarized data
    print("Saving summarized data...")
    save_summarized_data(df_with_summaries, output_data_path)

    print("Data summarization process completed!")
    return df_with_summaries


def main():
    """
    Example usage of the data summarization module.
    """
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define paths
    input_data_path = os.path.join(
        dir_path, "../data/processed/", "riskview_merged_data.json"
    )
    output_data_path = os.path.join(
        dir_path, "../data/processed/", "riskview_merged_data_with_summaries.json"
    )

    try:
        # Process summaries
        with get_openai_callback() as cb:
            df_with_summaries = process_summaries(input_data_path, output_data_path)
            print(cb)
            thb = cb.total_cost * 35
            print(f"total cost (THB): {thb}")

        print(f"Processing completed! Processed {len(df_with_summaries)} records")
        print(f"Summarized data saved to: {output_data_path}")
        print(
            "Added summary fields: risk_desc_summary, rootcause_summary, process_summary"
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please run data_standardizer.py first to create riskview_merged_data.json."
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
