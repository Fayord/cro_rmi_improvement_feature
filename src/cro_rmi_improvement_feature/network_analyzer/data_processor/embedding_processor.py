import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_standardized_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all standardized JSON files from the data directory.

    Args:
        data_dir: Path to the directory containing standardized JSON files

    Returns:
        List of all risk data records from all files
    """
    all_data = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all JSON files in the directory
    json_files = list(data_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    for json_file in json_files:
        print(f"Loading data from: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                all_data.extend(file_data)
                print(f"Loaded {len(file_data)} records from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"Total records loaded: {len(all_data)}")
    return all_data


def merge_all_data(standardized_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge all standardized data into a single DataFrame.

    Args:
        standardized_data: List of risk data records

    Returns:
        Merged DataFrame with all data
    """
    if not standardized_data:
        raise ValueError("No data to merge")

    # Convert to DataFrame
    df = pd.DataFrame(standardized_data)

    # Ensure all records have required fields
    required_fields = ["company", "risk", "risk_desc", "date_stamp"]
    missing_fields = [field for field in required_fields if field not in df.columns]

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    print(f"Merged data shape: {df.shape}")
    print(f"Companies in data: {df['company'].unique()}")
    print(f"Date stamps in data: {df['date_stamp'].unique()}")

    return df


def add_embeddings_to_data(
    df: pd.DataFrame,
    embedding_provider: str = "openai_large",
    max_embeddings: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add both embedding versions to each record in the DataFrame.

    Args:
        df: DataFrame containing risk data
        embedding_provider: Provider for embeddings ("openai_large", "openai_small", "sentence_transformers", etc.)
        max_embeddings: Maximum number of embeddings to generate (None for all)

    Returns:
        DataFrame with both embedding versions added
    """
    print(f"Adding embeddings using provider: {embedding_provider}")
    if max_embeddings:
        print(f"Limiting to {max_embeddings} embeddings")

    # Create text for embedding (combine relevant fields)
    def create_embedding_text_v1(row: pd.Series) -> str:
        """Create text for embedding version 1: risk, risk_desc, rootcause, rootcause_desc, process, process_desc."""
        text_parts = []

        # Check required columns exist
        required_columns_v1 = [
            "risk",
            "risk_desc",
            "rootcause",
            "rootcause_desc",
            "process",
            "process_desc",
        ]
        missing_columns = [col for col in required_columns_v1 if col not in row.index]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for embedding v1: {missing_columns}"
            )

        # Add risk name
        if row["risk"] is not None and str(row["risk"]).strip() != "":
            text_parts.append(f"Risk: {row['risk']}")

        # Add risk description
        if row["risk_desc"] is not None and str(row["risk_desc"]).strip() != "":
            text_parts.append(f"Description: {row['risk_desc']}")

        # Add root cause
        if row["rootcause"] is not None and str(row["rootcause"]).strip() != "":
            text_parts.append(f"Root Cause: {row['rootcause']}")

        # Add root cause description
        if (
            row["rootcause_desc"] is not None
            and str(row["rootcause_desc"]).strip() != ""
        ):
            text_parts.append(f"Root Cause Description: {row['rootcause_desc']}")

        # Add process
        if row["process"] is not None and str(row["process"]).strip() != "":
            text_parts.append(f"Process: {row['process']}")

        # Add process description
        if row["process_desc"] is not None and str(row["process_desc"]).strip() != "":
            text_parts.append(f"Process Description: {row['process_desc']}")

        return " | ".join(text_parts)

    def create_embedding_text_v2(row: pd.Series) -> str:
        """Create text for embedding version 2: risk, risk_desc + catalog risk_desc."""
        text_parts = []

        # Check required columns exist
        required_columns_v2 = ["risk", "risk_desc"]
        missing_columns = [col for col in required_columns_v2 if col not in row.index]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for embedding v2: {missing_columns}"
            )

        # Add risk name
        if row["risk"] is not None and str(row["risk"]).strip() != "":
            text_parts.append(f"Risk: {row['risk']}")

        # Add risk description
        if row["risk_desc"] is not None and str(row["risk_desc"]).strip() != "":
            text_parts.append(f"Description: {row['risk_desc']}")

        # Add catalog risk description if available (optional for v2)
        if (
            "risk_desc_catalog" in row.index
            and row["risk_desc_catalog"] is not None
            and str(row["risk_desc_catalog"]).strip() != ""
        ):
            text_parts.append(f"Catalog Description: {row['risk_desc_catalog']}")

        return " | ".join(text_parts)

    # Create embedding text for both versions
    df["embedding_text_v1"] = df.apply(create_embedding_text_v1, axis=1)
    df["embedding_text_v2"] = df.apply(create_embedding_text_v2, axis=1)

    # Import embedding providers
    import sys
    import os

    # Add the find_similar_risk directory to the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    find_similar_risk_path = os.path.join(current_dir, "../../../find_similar_risk")
    sys.path.insert(0, find_similar_risk_path)

    try:
        from embedding_providers import (
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
            from embedding_providers import (
                OpenAIEmbeddingProvider,
                SentenceTransformerProvider,
            )

            print(f"Successfully imported from alternative path: {alt_path}")
        except ImportError as e2:
            print(f"Alternative path also failed: {e2}")
            raise

    # Initialize embedding provider
    if embedding_provider == "openai_large":
        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-large")

        def get_openai_large_embedding(text: str, index: int) -> List[float]:
            """Get embedding from OpenAI large model."""
            if max_embeddings and index >= max_embeddings:
                return None
            if not text or text.strip() == "":
                return None
            try:
                embedding = provider.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                return None

        # Add embeddings for both versions
        print("Generating embeddings using OpenAI large model...")
        df["embedding_v1"] = [
            get_openai_large_embedding(text, idx)
            for idx, text in enumerate(df["embedding_text_v1"])
        ]
        df["embedding_v2"] = [
            get_openai_large_embedding(text, idx)
            for idx, text in enumerate(df["embedding_text_v2"])
        ]

    elif embedding_provider == "openai_small":
        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small")

        def get_openai_small_embedding(text: str, index: int) -> List[float]:
            """Get embedding from OpenAI small model."""
            if max_embeddings and index >= max_embeddings:
                return None
            if not text or text.strip() == "":
                return None
            try:
                embedding = provider.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                return None

        # Add embeddings for both versions
        print("Generating embeddings using OpenAI small model...")
        df["embedding_v1"] = [
            get_openai_small_embedding(text, idx)
            for idx, text in enumerate(df["embedding_text_v1"])
        ]
        df["embedding_v2"] = [
            get_openai_small_embedding(text, idx)
            for idx, text in enumerate(df["embedding_text_v2"])
        ]

    elif embedding_provider == "sentence_transformers":
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

        def get_sentence_transformer_embedding(text: str, index: int) -> List[float]:
            """Get embedding using sentence transformers."""
            if max_embeddings and index >= max_embeddings:
                return None
            if not text or text.strip() == "":
                return None
            try:
                embedding = provider.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                return None

        # Add embeddings for both versions
        print("Generating embeddings using sentence transformers...")
        df["embedding_v1"] = [
            get_sentence_transformer_embedding(text, idx)
            for idx, text in enumerate(df["embedding_text_v1"])
        ]
        df["embedding_v2"] = [
            get_sentence_transformer_embedding(text, idx)
            for idx, text in enumerate(df["embedding_text_v2"])
        ]

    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

    # Remove rows where embedding generation failed for either version
    initial_count = len(df)
    # Use a more robust filtering approach
    valid_embeddings = df["embedding_v1"].apply(lambda x: x is not None) & df[
        "embedding_v2"
    ].apply(lambda x: x is not None)
    df = df[valid_embeddings]
    final_count = len(df)

    print(f"Embeddings generated: {final_count}/{initial_count} records")

    return df


def save_embeddings_data(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame with embeddings to JSON file.

    Args:
        df: DataFrame with embeddings
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to list of dictionaries
    result_list = df.to_dict(orient="records")

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    result_json = convert_for_json(result_list)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"Embeddings data saved to: {output_path}")


def process_embeddings(
    standardized_data_dir: str,
    output_path: str,
    embedding_provider: str = "openai_large",
    max_embeddings: Optional[int] = None,
) -> pd.DataFrame:
    """
    Main function to process embeddings and add them to data.

    Args:
        standardized_data_dir: Directory containing standardized JSON files
        output_path: Path to save the embeddings data JSON file
        embedding_provider: Provider for embeddings
        max_embeddings: Maximum number of embeddings to generate (None for all)

    Returns:
        DataFrame with both embedding versions added
    """
    print("Starting embedding process...")

    # Load all standardized data
    print("Loading standardized data...")
    standardized_data = load_standardized_data(standardized_data_dir)

    # Merge all data
    print("Merging data...")
    merged_df = merge_all_data(standardized_data)

    # Add embeddings for both versions
    print("Adding embeddings for both versions...")
    df_with_embeddings = add_embeddings_to_data(
        merged_df, embedding_provider, max_embeddings
    )

    # Save embeddings data
    print("Saving embeddings data...")
    save_embeddings_data(df_with_embeddings, output_path)

    print("Embedding process completed!")
    return df_with_embeddings


# Flag to control sample run
IS_SAMPLE_RUN = True


def main():
    """
    Example usage of the embedding processor.
    """
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define paths
    standardized_data_dir = os.path.join(dir_path, "../data/standardized/")

    # Set output path based on sample run flag
    if IS_SAMPLE_RUN:
        output_path = os.path.join(
            dir_path, "../data/embeddings/", "sample_risk_data_with_embeddings.json"
        )
        print("Running in SAMPLE MODE - processing first 10 rows only")
    else:
        output_path = os.path.join(
            dir_path, "../data/embeddings/", "risk_data_with_embeddings.json"
        )
        print("Running in FULL MODE - processing all data")

    # Load all standardized data
    print("Loading standardized data...")
    standardized_data = load_standardized_data(standardized_data_dir)

    # Merge all data
    print("Merging data...")
    merged_df = merge_all_data(standardized_data)

    # Set max_embeddings for sample run
    max_embeddings = 10 if IS_SAMPLE_RUN else None
    if IS_SAMPLE_RUN:
        print("Running in SAMPLE MODE - limiting to 10 embeddings")
        print(f"Full data shape: {merged_df.shape}")

    # Add embeddings for both versions
    print("Adding embeddings for both versions...")
    df_with_embeddings = add_embeddings_to_data(
        merged_df, "openai_large", max_embeddings
    )

    # Save embeddings data
    print("Saving embeddings data...")
    save_embeddings_data(df_with_embeddings, output_path)

    print(f"Processing completed! Processed {len(df_with_embeddings)} records")
    print(f"Data saved to: {output_path}")
    print(f"Each record contains both embedding_v1 and embedding_v2")


if __name__ == "__main__":
    main()
