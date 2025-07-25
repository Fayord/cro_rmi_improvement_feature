"""
Embedding processing module for the Risk Network Visualization System.

This module handles the generation and management of embeddings for risk data.
"""

import os
import json
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional
import warnings
from pathlib import Path
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
import sys

# Import embedding providers
import sys
import os

# Add the find_similar_risk directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
find_similar_risk_path = os.path.join(current_dir, "../../../find_similar_risk")
sys.path.insert(0, find_similar_risk_path)
try:
    from embedding_providers import (  # type: ignore
        OpenAIEmbeddingProvider,
        SentenceTransformerProvider,
    )
except ImportError as e:
    print(f"Error importing embedding_providers: {e}")
    print(f"Tried to import from: {find_similar_risk_path}")
    print(f"Current sys.path: {sys.path}")

    # Try alternative path
    alt_path = os.path.join(current_dir, "../../find_similar_risk")
    sys.path.insert(0, alt_path)
    try:
        from embedding_providers import (  # type: ignore
            OpenAIEmbeddingProvider,
            SentenceTransformerProvider,
        )

        print(f"Successfully imported from alternative path: {alt_path}")
    except ImportError as e2:
        print(f"Alternative path also failed: {e2}")
        raise


def load_merged_data(input_path: str) -> pd.DataFrame:
    """
    Load merged data from JSON format.

    Args:
        input_path: Path to the JSON file

    Returns:
        DataFrame with merged data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Merged data file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded merged data: {df.shape}")
    return df


def add_embeddings_to_data(
    df: pd.DataFrame,
    embedding_provider: str = "openai_large",
    sample_rows_per_company: Optional[int] = None,  # New parameter
) -> pd.DataFrame:
    """
    Add both embedding versions to each record in the DataFrame.

    Args:
        df: DataFrame containing risk data
        embedding_provider: Provider for embeddings ("openai_large", "openai_small", "sentence_transformers", etc.)
        sample_rows_per_company: Maximum number of embeddings to generate per company (None for all)

    Returns:
        DataFrame with both embedding versions added
    """
    print(f"Adding embeddings using provider: {embedding_provider}")
    if sample_rows_per_company:
        print(f"Limiting to {sample_rows_per_company} embeddings per company")

    # If sample_rows_per_company is set, sample the DataFrame
    if sample_rows_per_company and "company" in df.columns:
        print(f"Sampling data to {sample_rows_per_company} rows per company...")
        df = df.groupby("company").head(sample_rows_per_company).reset_index(drop=True)
        print(f"Sampled DataFrame shape: {df.shape}")
    elif sample_rows_per_company and "company" not in df.columns:
        print(
            "Warning: 'company' column not found for per-company sampling. Limiting globally."
        )
        df = df.head(sample_rows_per_company).reset_index(drop=True)

    # Create text for embedding (combine relevant fields)
    def create_embedding_text_raw_user_data(row: pd.Series) -> str:
        """Create text for embedding: risk, risk_desc, rootcause_data, process_data (raw user data)."""
        text_parts = []

        # Check required columns exist
        required_columns = [
            "risk",
            "risk_desc",
            "rootcause_data",
            "process_data",
        ]
        missing_columns = [col for col in required_columns if col not in row.index]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for raw user data embedding: {missing_columns}"
            )

        # Add risk name
        if row["risk"] is not None and str(row["risk"]).strip() != "":
            text_parts.append(f"Risk: {row['risk']}")

        # Add risk description
        if row["risk_desc"] is not None and str(row["risk_desc"]).strip() != "":
            text_parts.append(f"Description: {row['risk_desc']}")

        # Add root cause data (combined field)
        if (
            row["rootcause_data"] is not None
            and str(row["rootcause_data"]).strip() != ""
        ):
            text_parts.append(f"Root Cause: {row['rootcause_data']}")

        # Add process data (combined field)
        if row["process_data"] is not None and str(row["process_data"]).strip() != "":
            text_parts.append(f"Process: {row['process_data']}")

        return " | ".join(text_parts)

    def create_embedding_text_risk_desc_catalog(row: pd.Series) -> str:
        """Create text for embedding: risk, risk_desc + catalog risk_desc."""
        text_parts = []

        # Check required columns exist
        required_columns = ["risk", "risk_desc"]
        missing_columns = [col for col in required_columns if col not in row.index]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for risk_desc_catalog embedding: {missing_columns}"
            )

        # Add risk name
        if row["risk"] is not None and str(row["risk"]).strip() != "":
            text_parts.append(f"Risk: {row['risk']}")

        # Add risk description
        if row["risk_desc"] is not None and str(row["risk_desc"]).strip() != "":
            text_parts.append(f"Description: {row['risk_desc']}")

        # Add catalog risk description if available
        if (
            "risk_desc_catalog" in row.index
            and row["risk_desc_catalog"] is not None
            and str(row["risk_desc_catalog"]).strip() != ""
        ):
            text_parts.append(f"Catalog Description: {row['risk_desc_catalog']}")

        return " | ".join(text_parts)

    def create_embedding_text_summary_user_data(row: pd.Series) -> str:
        """Create text for embedding: risk + LLM summaries."""
        text_parts = []

        # Add risk name
        if row["risk"] is not None and str(row["risk"]).strip() != "":
            text_parts.append(f"Risk: {row['risk']}")

        # Add summary fields if they exist
        if (
            "risk_desc_summary" in row.index
            and row["risk_desc_summary"] is not None
            and str(row["risk_desc_summary"]).strip() != ""
        ):
            text_parts.append(f"Risk Summary: {row['risk_desc_summary']}")

        if (
            "rootcause_summary" in row.index
            and row["rootcause_summary"] is not None
            and str(row["rootcause_summary"]).strip() != ""
        ):
            text_parts.append(f"Root Cause Summary: {row['rootcause_summary']}")

        if (
            "process_summary" in row.index
            and row["process_summary"] is not None
            and str(row["process_summary"]).strip() != ""
        ):
            text_parts.append(f"Process Summary: {row['process_summary']}")

        return " | ".join(text_parts)

    # Create embedding text for all three versions
    df["embedding_text_raw_user_data"] = df.apply(
        create_embedding_text_raw_user_data, axis=1
    )
    df["embedding_text_risk_desc_catalog"] = df.apply(
        create_embedding_text_risk_desc_catalog, axis=1
    )
    df["embedding_text_summary_user_data"] = df.apply(
        create_embedding_text_summary_user_data, axis=1
    )

    # Initialize embedding provider
    if embedding_provider == "openai_large":
        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-large")

        def get_openai_large_embedding(text: str, index: int) -> List[float]:
            """Get embedding from OpenAI large model."""
            if not text or text.strip() == "":
                return None
            try:
                embedding = provider.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                return None

        # Add embeddings for all three versions
        print("Generating embeddings using OpenAI large model...")

        # Generate embeddings with progress bars
        df["embedding_raw_user_data"] = [
            get_openai_large_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_raw_user_data"]),
                total=len(df),
                desc="Raw user data embeddings",
                unit="embedding",
            )
        ]
        df["embedding_risk_desc_catalog"] = [
            get_openai_large_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_risk_desc_catalog"]),
                total=len(df),
                desc="Risk desc catalog embeddings",
                unit="embedding",
            )
        ]
        df["embedding_summary_user_data"] = [
            get_openai_large_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_summary_user_data"]),
                total=len(df),
                desc="Summary user data embeddings",
                unit="embedding",
            )
        ]

    elif embedding_provider == "openai_small":
        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small")

        def get_openai_small_embedding(text: str, index: int) -> List[float]:
            """Get embedding from OpenAI small model."""
            if not text or text.strip() == "":
                return None
            try:
                embedding = provider.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                return None

        # Add embeddings for all three versions
        print("Generating embeddings using OpenAI small model...")

        # Generate embeddings with progress bars
        df["embedding_raw_user_data"] = [
            get_openai_small_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_raw_user_data"]),
                total=len(df),
                desc="Raw user data embeddings",
                unit="embedding",
            )
        ]
        df["embedding_risk_desc_catalog"] = [
            get_openai_small_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_risk_desc_catalog"]),
                total=len(df),
                desc="Risk desc catalog embeddings",
                unit="embedding",
            )
        ]
        df["embedding_summary_user_data"] = [
            get_openai_small_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_summary_user_data"]),
                total=len(df),
                desc="Summary user data embeddings",
                unit="embedding",
            )
        ]

    elif embedding_provider == "sentence_transformers":
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

        def get_sentence_transformer_embedding(text: str, index: int) -> List[float]:
            """Get embedding using sentence transformers."""
            if not text or text.strip() == "":
                return None
            try:
                embedding = provider.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                return None

        # Add embeddings for all three versions
        print("Generating embeddings using sentence transformers...")

        # Generate embeddings with progress bars
        df["embedding_raw_user_data"] = [
            get_sentence_transformer_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_raw_user_data"]),
                total=len(df),
                desc="Raw user data embeddings",
                unit="embedding",
            )
        ]
        df["embedding_risk_desc_catalog"] = [
            get_sentence_transformer_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_risk_desc_catalog"]),
                total=len(df),
                desc="Risk desc catalog embeddings",
                unit="embedding",
            )
        ]
        df["embedding_summary_user_data"] = [
            get_sentence_transformer_embedding(text, idx)
            for idx, text in tqdm(
                enumerate(df["embedding_text_summary_user_data"]),
                total=len(df),
                desc="Summary user data embeddings",
                unit="embedding",
            )
        ]

    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

    # Remove rows where embedding generation failed for any version
    initial_count = len(df)
    # Use a more robust filtering approach
    valid_embeddings = (
        df["embedding_raw_user_data"].apply(lambda x: x is not None)
        & df["embedding_risk_desc_catalog"].apply(lambda x: x is not None)
        & df["embedding_summary_user_data"].apply(lambda x: x is not None)
    )
    df = df[valid_embeddings]
    final_count = len(df)

    print(f"Embeddings generated: {final_count}/{initial_count} records")
    print(
        "Generated embeddings for: raw_user_data, risk_desc_catalog, summary_user_data"
    )

    return df


def save_embeddings_data(
    df: pd.DataFrame, output_path: str, is_sample_run: bool = False
):
    """
    Save DataFrame with embeddings to pickle format (or JSON for sample runs).

    Args:
        df: DataFrame with embeddings
        output_path: Path to save the file
        is_sample_run: If True, save as JSON for easier debugging
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if is_sample_run:
        # For sample runs, save as JSON for easier debugging
        # Convert embeddings to lists for JSON serialization
        df_copy = df.copy()

        # Convert embedding columns to lists if they exist
        embedding_columns = [
            col for col in df_copy.columns if col.startswith("embedding_")
        ]
        for col in embedding_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )

        # Save as JSON
        data_to_save = df_copy.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"Sample embeddings data saved to: {output_path} (JSON format)")
    else:
        # Save as pickle for better compatibility with mixed data types and embeddings
        with open(output_path, "wb") as f:
            pickle.dump(df, f)
        print(f"Embeddings data saved to: {output_path} (pickle format)")


def process_embeddings(
    merged_data_path: str,
    embeddings_output_path: str,
    embedding_provider: str = "openai_large",
    sample_rows_per_company: Optional[int] = None,  # New parameter
    is_sample_run: bool = False,
) -> pd.DataFrame:
    """
    Process embeddings and add them to merged data.

    Args:
        merged_data_path: Path to the merged data JSON file
        embeddings_output_path: Path to save the embeddings data file
        embedding_provider: Provider for embeddings
        sample_rows_per_company: Maximum number of rows to process per company (None for all)
        is_sample_run: If True, save embeddings as JSON for easier debugging

    Returns:
        DataFrame with both embedding versions added
    """
    print("Starting embedding process...")

    # Load merged data
    print("Loading merged data...")
    merged_df = load_merged_data(merged_data_path)

    # Add embeddings for both versions
    print("Adding embeddings for both versions...")
    df_with_embeddings = add_embeddings_to_data(
        merged_df, embedding_provider, sample_rows_per_company  # Pass new parameter
    )

    # Save embeddings data
    print("Saving embeddings data...")
    save_embeddings_data(
        df_with_embeddings, embeddings_output_path, is_sample_run=is_sample_run
    )

    print("Embedding process completed!")
    return df_with_embeddings


# Flag to control sample run
IS_SAMPLE_RUN = False
SAMPLE_ROWS_PER_COMPANY = 10  # Default to 10 rows per company


def main():
    """
    Example usage of the embedding process module.
    """
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Hardcode data type as riskview
    data_type = "riskview"
    print(f"Processing data type: {data_type}")

    # Define paths - load from summarized data
    merged_data_path = os.path.join(
        dir_path, "../data/processed/", f"{data_type}_merged_data_with_summaries.json"
    )

    # Set output paths based on sample run flag
    if IS_SAMPLE_RUN:
        embeddings_output_path = os.path.join(
            dir_path,
            "../data/embeddings/",
            f"{data_type}_sample_data_with_embeddings.json",
        )
        print(
            f"Running in SAMPLE MODE - limiting to {SAMPLE_ROWS_PER_COMPANY} rows per company"
        )
        current_sample_rows_per_company = SAMPLE_ROWS_PER_COMPANY
    else:
        embeddings_output_path = os.path.join(
            dir_path, "../data/embeddings/", f"{data_type}_data_with_embeddings.pkl"
        )
        print("Running in FULL MODE - processing all data")
        current_sample_rows_per_company = None

    try:
        # Process embeddings
        df_with_embeddings = process_embeddings(
            merged_data_path,
            embeddings_output_path,
            "openai_large",
            sample_rows_per_company=current_sample_rows_per_company,
            is_sample_run=IS_SAMPLE_RUN,
        )

        print(f"Processing completed! Processed {len(df_with_embeddings)} records")
        # show number of records per company
        print(
            f"Number of records per company: {df_with_embeddings['company'].value_counts()}"
        )
        print(f"Embeddings data saved to: {embeddings_output_path}")
        print(
            f"Each record contains embeddings for: raw_user_data, risk_desc_catalog, summary_user_data"
        )
        print("Embeddings include original data + LLM summaries")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            f"Please run summarize_data.py first to create {data_type}_merged_data_with_summaries.json."
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
