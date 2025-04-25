from typing import List, Dict, Union, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
from main import find_similar_sentences_batch, cosine_distance, euclidean_distance
from embedding_providers import BaseEmbeddingProvider
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm


class CompareResultDict(BaseModel):
    input_sentence: str
    target_sentence: str
    candidate_sentences: List[str]
    embedded_model: str
    distance_method: str
    sorted_similar_sentences: List[str]
    sorted_similar_sentences_indices: List[int]
    target_order_in_sorted_similar_sentences: int  # starts with 1


def compare_target_sentence_rankings(
    input_sentences: List[str],
    candidate_sets: Dict[str, Dict[str, Union[str, List[str]]]],
    embedding_models: Dict[str, BaseEmbeddingProvider],
    distance_functions: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> List[dict]:
    """
    Compare different embedding models and distance metrics.
    """
    results = []

    # Validate target_sentences length matches input_sentences
    for set_name, candidate_set in candidate_sets.items():
        if len(candidate_set["target_sentences"]) != len(input_sentences):
            raise ValueError(
                f"Number of target sentences ({len(candidate_set['target_sentences'])}) "
                f"in set '{set_name}' does not match number of input sentences ({len(input_sentences)})"
            )

        # Validate all target sentences are in candidate sentences
        for target_sentence in candidate_set["target_sentences"]:
            if target_sentence not in candidate_set["candidate_sentences"]:
                raise ValueError(
                    f"Target sentence '{target_sentence}' not found in candidate set {set_name}"
                )
    total_iterations = (
        len(input_sentences)
        * len(candidate_sets)
        * len(embedding_models)
        * len(distance_functions)
    )

    # Create progress bar
    pbar = tqdm(total=total_iterations, desc="Processing comparisons")
    for idx, input_sentence in enumerate(input_sentences):  # this line add tqdm
        for set_name, candidate_set in candidate_sets.items():
            candidate_sentences = candidate_set["candidate_sentences"]
            target_sentence = candidate_set["target_sentences"][idx]

            for model_name, model in embedding_models.items():
                # Generate embeddings using get_embedding method
                input_embedding = model.get_embedding(input_sentence)
                candidate_embeddings = np.array(
                    [model.get_embedding(sent) for sent in candidate_sentences]
                )

                for dist_name, dist_func in distance_functions.items():
                    batch_results = find_similar_sentences_batch(
                        [input_sentence], candidate_sentences, dist_func, model
                    )

                    sorted_sentences = batch_results[0]["similar_sentences"]
                    sorted_indices = batch_results[0]["similar_indices"]

                    try:
                        target_position = sorted_sentences.index(target_sentence) + 1
                    except ValueError:
                        raise ValueError(
                            f"Target sentence '{target_sentence}' not found in sorted results for "
                            f"model: {model_name}, distance: {dist_name}"
                        )
                    result = CompareResultDict(
                        input_sentence=input_sentence,
                        target_sentence=target_sentence,
                        candidate_sentences=candidate_sentences,
                        embedded_model=model_name,
                        distance_method=dist_name,
                        sorted_similar_sentences=sorted_sentences,
                        sorted_similar_sentences_indices=sorted_indices,
                        target_order_in_sorted_similar_sentences=target_position,
                    )
                    result_dict = result.model_dump(mode="json")
                    results.append(result_dict)
                    pbar.update(1)
    pbar.close()

    return results


if __name__ == "__main__":
    # Example usage
    input_sentences = [
        "Market risk affects investment returns",
        "Cybersecurity poses significant threats",
        "Credit risk in lending operations",
    ]

    old_candidate_sets = {
        "candidate_set_1": {
            "candidate_sentences": [
                "Market fluctuations impact returns",
                "Weather is nice today",
                "Investment returns affected by market",
                "Cybersecurity threats are increasing",
            ],
            "target_sentence": "Market fluctuations impact returns",
        },
        "candidate_set_1": {
            "candidate_sentences": [
                "Credit risk assessment in banking",
                "Market volatility affects investments",
                "Cyber attacks on organizations",
                "Risk management in finance",
            ],
            "target_sentence": "Market volatility affects investments",
        },
        "candidate_set_1": {
            "candidate_sentences": [
                "Financial market volatility",
                "Credit risk evaluation methods",
                "Information security threats",
                "Enterprise risk management",
            ],
            "target_sentence": "Financial market volatility",
        },
    }
    candidate_sets = {
        "candidate_set_1": {
            "candidate_sentences": [
                "Market fluctuations impact returns",
                "Weather is nice today",
                "Investment returns affected by market",
                "Cybersecurity threats are increasing",
            ],
            "target_sentences": [
                "Market fluctuations impact returns",
                "Market fluctuations impact returns",
                "Investment returns affected by market",
            ],
        },
        "candidate_set_2": {
            "candidate_sentences": [
                "Credit risk assessment in banking",
                "Market volatility affects investments",
                "Cyber attacks on organizations",
                "Risk management in finance",
            ],
            "target_sentences": [
                "Market volatility affects investments",
                "Market volatility affects investments",
                "Cyber attacks on organizations",
            ],
        },
        "candidate_set_3": {
            "candidate_sentences": [
                "Financial market volatility",
                "Credit risk evaluation methods",
                "Information security threats",
                "Enterprise risk management",
            ],
            "target_sentences": [
                "Financial market volatility",
                "Financial market volatility",
                "Information security threats",
            ],
        },
    }

    # Initialize models
    embedding_models = {
        "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
        "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    }

    # Initialize distance functions
    distance_functions = {"cosine": cosine_distance, "euclidean": euclidean_distance}

    comparison_results = compare_target_sentence_rankings(
        input_sentences, candidate_sets, embedding_models, distance_functions
    )
