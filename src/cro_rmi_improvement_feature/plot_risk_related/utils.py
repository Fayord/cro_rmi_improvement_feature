import random
from typing import Any, Dict, List, Optional, Literal
import os
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_community.cache import SQLiteCache
from plot_network_defaut_value import (
    EDGE_SIZE_MULTIPLIER,
    edge_rgb_color_list,
    rgb_color_list,
)
import pickle

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np
from collections import Counter

# import re
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


def generate_network_from_real_data(data_list, selected_checklist_values=None):
    """
    data_list: list of dicts, each dict contains:
        - "risk": str
        - "embedding_risk": list or np.array
        - "risk_desc": str
        - "embedding_risk_desc": list or np.array
    """
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    except Exception:
        dir_path = os.getcwd()
    edge_relationships_data_path = f"{dir_path}/edge_relationships.pkl"
    if os.path.exists(edge_relationships_data_path):
        edge_relationships_data = pickle.load(open(edge_relationships_data_path, "rb"))
    else:
        edge_relationships_data = {}
    node_size_multiplier = 10
    number_of_scales = 3
    node_proportion_list = [65, 30, 5]
    node_size_counter = Counter()
    assert len(node_proportion_list) == number_of_scales
    # node_size_list = [1, 30, 60]

    node_size_list = [1, 50, 120]
    assert len(node_size_list) == number_of_scales
    nodes = []
    edges = []
    line_weight_list = []
    # print checkbox value in this function
    selected_checkbox_value_list = []
    if selected_checklist_values is None:
        selected_checklist_values = []
    for selected_checklist_value in selected_checklist_values:
        selected_checkbox_value_list.append(selected_checklist_value)

    # Create edges based on embedding distance (Euclidean)
    # cal the distance between each pair of risks first
    stored_distances = {}
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            emb1 = np.array(
                data_list[i][
                    tuple(sorted(["risk", "embedding"] + selected_checkbox_value_list))
                ]
            )
            emb2 = np.array(
                data_list[j][
                    tuple(sorted(["risk", "embedding"] + selected_checkbox_value_list))
                ]
            )
            distance = np.linalg.norm(emb1 - emb2)
            distance = 1 / distance * 100
            line_weight_list.append(distance)
            stored_distances[(i, j)] = distance

    min_dist = min(line_weight_list)
    max_dist = max(line_weight_list)
    avg_dist = np.mean(line_weight_list)
    print(f"{avg_dist=}")

    edge_boudaries = find_equal_count_boundaries(line_weight_list, number_of_scales)
    # then create edges
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            raw_weight = stored_distances[(i, j)]
            # display_weight is seperated into 3 scales based on the min_dist and max_dist
            level = get_level_from_boundaries(edge_boudaries, raw_weight)
            display_weight = level * EDGE_SIZE_MULTIPLIER
            edge_color = edge_rgb_color_list[level - 1]
            edge_relation_i_j = edge_relationships_data.get(
                tuple((data_list[i]["risk"], data_list[j]["risk"])), None
            )
            edge_relation_j_i = edge_relationships_data.get(
                tuple((data_list[j]["risk"], data_list[i]["risk"])), None
            )
            if edge_relation_i_j is None and edge_relation_j_i is None:
                # no relationship
                edge_relation = [0, 0]
                edge_relation_reason = ""
            elif edge_relation_i_j is not None and edge_relation_j_i is None:
                edge_relation = edge_relation_i_j["direction"]
                edge_relation_reason = edge_relation_i_j["reason"]
                if edge_relation[0] == 1:
                    i, j = i, j
                elif edge_relation[0] == -1:
                    i, j = j, i
            elif edge_relation_i_j is None and edge_relation_j_i is not None:
                edge_relation = edge_relation_j_i["direction"]
                edge_relation_reason = edge_relation_j_i["reason"]
                if edge_relation[0] == 1:
                    i, j = j, i
                elif edge_relation[0] == -1:
                    i, j = i, j

            else:
                raise ValueError("it should have 1 None")
            # print(f"{edge_relation_i_j=},{edge_relation_j_i=}")
            edges.append(
                {
                    "data": {
                        "source": f"risk_{i}",
                        "target": f"risk_{j}",
                        "weight": display_weight,
                        "raw_weight": raw_weight,
                        "color": edge_color,
                        "arrow_weight": (
                            "none" if edge_relation[0] == 0 else "triangle"
                        ),
                        "do_not_cal_weight": False,
                        "edge_relation_reason": edge_relation_reason,
                    }
                }
            )
            if edge_relation[1] == -1:
                edges.append(
                    {
                        "data": {
                            "source": f"risk_{j}",
                            "target": f"risk_{i}",
                            "weight": display_weight,
                            "raw_weight": raw_weight,
                            "color": edge_color,
                            "arrow_weight": "triangle",
                            "do_not_cal_weight": True,
                            "edge_relation_reason": edge_relation_reason,
                        }
                    }
                )

    # Calculate raw_size for each node (sum of all connected edge weights)
    node_raw_sizes = [0.0 for _ in range(len(data_list))]

    for edge in edges:
        if edge["data"]["do_not_cal_weight"]:
            continue
        src = int(edge["data"]["source"].split("_")[1])
        tgt = int(edge["data"]["target"].split("_")[1])
        w = edge["data"]["raw_weight"]
        node_raw_sizes[src] += w
        node_raw_sizes[tgt] += w

    node_boudaries = find_proportional_count_boundaries(
        node_raw_sizes, node_proportion_list
    )
    # Scale node sizes for display (between 20 and 100)
    min_raw = min(node_raw_sizes)
    max_raw = max(node_raw_sizes)

    def scale_size(raw):
        if max_raw == min_raw:
            return 60  # fallback if all are equal
        return 20 + (raw - min_raw) / (max_raw - min_raw) * (100 - 20)

    all_risk_cat = []
    for data in data_list:
        risk_cat = data["risk_cat"]
        if risk_cat not in all_risk_cat:
            all_risk_cat.append(risk_cat)
    all_risk_cat.sort()
    # Create nodes for each risk
    for idx, data in enumerate(data_list):
        raw_size = node_raw_sizes[idx]
        level = get_level_from_boundaries(node_boudaries, raw_size)
        node_size_counter[level] += 1
        display_size = node_size_list[level - 1]
        risk_cat = data["risk_cat"]
        risk_cat_color = rgb_color_list[all_risk_cat.index(risk_cat)]
        nodes.append(
            {
                "data": {
                    "id": f"risk_{idx}",
                    "label": data["risk"],
                    "raw_size": raw_size,
                    "size_level": level,
                    "size": display_size,
                    "color": risk_cat_color,
                    "risk_level": data["risk_level"],
                    "story": data.get("story", ""),
                },
                "position": {
                    "x": random.uniform(100, 700),
                    "y": random.uniform(100, 700),
                },
            }
        )
    print(f"{node_size_counter=}")
    # Return nodes, edges, line_weight_list, and total number of edges
    return nodes + edges, line_weight_list, len(edges)


# Initial elements for the default company
def get_elements_for_company(data, company, selected_checklist_values):
    filtered = [item for item in data if item["company"] == company]
    if filtered:
        # Capture total_edges from the return value
        elements, line_weights, total_edges = generate_network_from_real_data(
            filtered, selected_checklist_values
        )
        return elements, line_weights, total_edges
    else:
        return [], [], 0  # Return 0 total edges if no data


# --- New function to calculate custom pyramid layout positions ---
def calculate_pyramid_layout(
    selected_node_id, primary_neighbors, secondary_neighbors, subgraph_elements
):
    # Define base positions and spacing
    center_x = 500
    top_y = 100
    primary_y_base = 175
    secondary_y_base = 250
    horizontal_spacing = 150
    vertical_zigzag_offset = 10  # How much to offset vertically for zigzag

    # Create a dictionary to store calculated positions by node ID
    positions = {}

    # Position the selected node at the top center
    positions[selected_node_id] = {"x": center_x, "y": top_y}

    # Position primary neighbors
    num_primary = len(primary_neighbors)
    if num_primary > 0:
        # Calculate starting x position to center the primary nodes
        primary_start_x = center_x - (num_primary - 1) * horizontal_spacing / 2
        for i, node_id in enumerate(primary_neighbors):
            x = primary_start_x + i * horizontal_spacing
            # Apply zigzag vertical offset
            y = primary_y_base + (vertical_zigzag_offset if i % 2 == 1 else 0)
            positions[node_id] = {"x": x, "y": y}

    # Position secondary neighbors
    num_secondary = len(secondary_neighbors)
    if num_secondary > 0:
        # Calculate starting x position to center the secondary nodes
        secondary_start_x = center_x - (num_secondary - 1) * horizontal_spacing / 2
        for i, node_id in enumerate(secondary_neighbors):
            x = secondary_start_x + i * horizontal_spacing
            # Apply zigzag vertical offset (use a different offset or pattern if desired)
            y = secondary_y_base + (
                vertical_zigzag_offset if i % 2 == 0 else 0
            )  # Alternate zigzag pattern
            positions[node_id] = {"x": x, "y": y}

    # Update the position data for each node in the subgraph elements
    updated_elements = []
    for el in subgraph_elements:
        if "source" not in el.get("data", {}):  # It's a node
            node_id = el["data"]["id"]
            if node_id in positions:
                el["position"] = positions[node_id]
        updated_elements.append(el)

    return updated_elements


# New function to generate dynamic stylesheet
def generate_dynamic_stylesheet(
    bezier_stylesheet,
    bezier_step_size,
    bezier_weight,
    node_highlight_selector_risk_level1,
    node_highlight_selector_risk_level2,
    node_highlight_selector_risk_level3,
    node_highlight_selector_risk_level4,
):
    dynamic_bezier_stylesheet = [
        {
            "selector": "node",
            "style": bezier_stylesheet[0]["style"],
        },
        node_highlight_selector_risk_level1,
        node_highlight_selector_risk_level2,
        node_highlight_selector_risk_level3,
        node_highlight_selector_risk_level4,
        {
            "selector": "edge",
            "style": {
                "curve-style": "unbundled-bezier",
                "control-point-step-size": bezier_step_size,
                "control-point-weight": bezier_weight,
                "opacity": 0.6,
                "line-color": "data(color)",
                "width": "mapData(weight, 0, 20, 1, 8)",
                "overlay-padding": "3px",
                "content": "data(weight)",
                "font-size": "0px",
                "text-valign": "center",
                "text-halign": "center",
                # --- Add arrow properties here ---
                "target-arrow-shape": "data(arrow_weight)",
                "target-arrow-color": "data(color)",  # Make the arrow color match the edge color
                "arrow-scale": "1",  # Adjust arrow size if needed (default is 1)
                # "source-arrow-shape": "circle", # Example for source arrow
                # "source-arrow-color": "blue",
                # --- End of arrow properties ---
            },
        },
    ]
    return dynamic_bezier_stylesheet


# --- Helper function to find primary and secondary neighbors ---
def find_neighbors(selected_node_id, all_elements):
    primary_neighbors = set()
    secondary_neighbors = set()
    edges_to_primary = []

    # Find primary neighbors and edges
    for el in all_elements:
        if "source" in el.get("data", {}):  # It's an edge
            source = el["data"]["source"]
            target = el["data"]["target"]
            if source == selected_node_id:
                primary_neighbors.add(target)
                edges_to_primary.append(el)
            elif target == selected_node_id:
                primary_neighbors.add(source)
                edges_to_primary.append(el)

    # Find secondary neighbors
    for el in all_elements:
        if "source" in el.get("data", {}):  # It's an edge
            source = el["data"]["source"]
            target = el["data"]["target"]
            # If the edge connects a primary neighbor to another node
            if (
                source in primary_neighbors
                and target != selected_node_id
                and target not in primary_neighbors
            ):
                secondary_neighbors.add(target)
            elif (
                target in primary_neighbors
                and source != selected_node_id
                and source not in primary_neighbors
            ):
                secondary_neighbors.add(source)

    # Get node elements for selected, primary, and secondary neighbors
    subgraph_nodes = []
    all_neighbor_ids = primary_neighbors.union(secondary_neighbors)
    all_neighbor_ids.add(selected_node_id)  # Include the selected node itself

    for el in all_elements:
        if "source" not in el.get("data", {}):  # It's a node
            if el["data"]["id"] in all_neighbor_ids:
                subgraph_nodes.append(el)

    # Get edges within the subgraph (between selected, primary, and secondary nodes)
    subgraph_edges = []
    for el in all_elements:
        if "source" in el.get("data", {}):  # It's an edge
            source = el["data"]["source"]
            target = el["data"]["target"]
            if source in all_neighbor_ids and target in all_neighbor_ids:
                subgraph_edges.append(el)

    return (
        list(primary_neighbors),
        list(secondary_neighbors),
        subgraph_nodes + subgraph_edges,
    )


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
