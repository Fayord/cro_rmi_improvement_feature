#!/usr/bin/env python3
"""
Test different saving methods for data with list embeddings.

This script loads sample embedding data, duplicates it 50x, and tests
various saving methods with timing measurements.
"""

import os
import json
import pandas as pd
import pickle
import time
from pathlib import Path


def load_sample_data():
    """Load the sample embedding data."""
    data_path = "/Users/ford/Documents/coding_trae/cro_rmi_improvement_feature/src/cro_rmi_improvement_feature/network_analyzer/data/embeddings/riskview_sample_data_with_embeddings.json"

    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        return None

    print(f"Loading data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data


def duplicate_data(data, multiplier=50):
    """Duplicate the data to simulate larger dataset."""
    print(f"Duplicating data {multiplier}x...")

    duplicated_data = []
    for i in range(multiplier):
        for record in data:
            # Create a copy with unique identifier
            new_record = record.copy()
            new_record["duplicate_id"] = i
            duplicated_data.append(new_record)

    print(
        f"Created {len(duplicated_data)} records (original: {len(data)} x {multiplier})"
    )
    return duplicated_data


def test_dict_save_load(data, filename):
    """Test saving/loading as dictionary with pickle."""
    print(f"\n=== Testing Dict Save/Load: {filename} ===")

    # Save as dict
    start_time = time.time()
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    save_time = time.time() - start_time
    file_size = os.path.getsize(filename)

    print(f"Save time: {save_time:.4f}s")
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    # Load as dict
    start_time = time.time()
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    load_time = time.time() - start_time

    print(f"Load time: {load_time:.4f}s")
    print(f"Records loaded: {len(loaded_data)}")

    return save_time, load_time, file_size


def test_dataframe_save_load(df, filename, method="pickle"):
    """Test saving/loading as DataFrame."""
    print(f"\n=== Testing DataFrame Save/Load: {filename} ({method}) ===")

    # Save DataFrame
    start_time = time.time()
    if method == "pickle":
        df.to_pickle(filename)
    elif method == "parquet":
        df.to_parquet(filename)
    save_time = time.time() - start_time
    file_size = os.path.getsize(filename)

    print(f"Save time: {save_time:.4f}s")
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    # Load DataFrame
    start_time = time.time()
    if method == "pickle":
        loaded_df = pd.read_pickle(filename)
    elif method == "parquet":
        loaded_df = pd.read_parquet(filename)
    load_time = time.time() - start_time

    print(f"Load time: {load_time:.4f}s")
    print(f"DataFrame shape: {loaded_df.shape}")

    return save_time, load_time, file_size


def analyze_data_structure(data):
    """Analyze the structure of the data."""
    print("\n=== Data Structure Analysis ===")

    if not data:
        print("No data to analyze")
        return

    sample_record = data[0]
    print(f"Sample record keys: {list(sample_record.keys())}")

    # Check for list data
    list_columns = []
    for key, value in sample_record.items():
        if isinstance(value, list):
            list_columns.append(key)
            print(
                f"List column '{key}': {len(value)} items, type: {type(value[0]) if value else 'empty'}"
            )

    print(f"Total list columns: {len(list_columns)}")

    # Check DataFrame conversion
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")

    return df


def main():
    """Main test function."""
    print("🚀 Starting Save/Load Performance Test")
    print("=" * 60)

    # Load sample data
    data = load_sample_data()
    if not data:
        return

    # Analyze original data
    df = analyze_data_structure(data)

    # Duplicate data
    duplicated_data = duplicate_data(data, multiplier=5000)

    # Create output directory
    dir_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(dir_path, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Test results storage
    results = []

    # Test 1: Dict with pickle
    save_time, load_time, file_size = test_dict_save_load(
        duplicated_data, os.path.join(output_dir, "test_dict.pkl")
    )
    results.append(
        {
            "method": "Dict (pickle)",
            "save_time": save_time,
            "load_time": load_time,
            "file_size": file_size,
        }
    )

    # Test 2: DataFrame with pickle
    save_time, load_time, file_size = test_dataframe_save_load(
        df, os.path.join(output_dir, "test_df.pkl"), "pickle"
    )
    results.append(
        {
            "method": "DataFrame (pickle)",
            "save_time": save_time,
            "load_time": load_time,
            "file_size": file_size,
        }
    )

    # Test 3: DataFrame with parquet
    try:
        save_time, load_time, file_size = test_dataframe_save_load(
            df, os.path.join(output_dir, "test_df.parquet"), "parquet"
        )
        results.append(
            {
                "method": "DataFrame (parquet)",
                "save_time": save_time,
                "load_time": load_time,
                "file_size": file_size,
            }
        )
    except Exception as e:
        print(f"Parquet test failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result['method']}:")
        print(f"  Save time: {result['save_time']:.4f}s")
        print(f"  Load time: {result['load_time']:.4f}s")
        print(f"  File size: {result['file_size']/1024/1024:.2f} MB")

    # Find fastest methods
    fastest_save = min(results, key=lambda x: x["save_time"])
    fastest_load = min(results, key=lambda x: x["load_time"])
    smallest_file = min(results, key=lambda x: x["file_size"])

    print(f"\n🏆 WINNERS:")
    print(
        f"  Fastest save: {fastest_save['method']} ({fastest_save['save_time']:.4f}s)"
    )
    print(
        f"  Fastest load: {fastest_load['method']} ({fastest_load['load_time']:.4f}s)"
    )
    print(
        f"  Smallest file: {smallest_file['method']} ({smallest_file['file_size']/1024/1024:.2f} MB)"
    )

    print(f"\n📁 Test files saved in: {output_dir}/")


if __name__ == "__main__":
    main()
