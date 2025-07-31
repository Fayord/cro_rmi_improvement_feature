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

dir_path = os.path.dirname(os.path.abspath(__file__))

set_llm_cache(SQLiteCache(database_path=f"{dir_path}/.langchain.db"))

env_path = os.path.join(dir_path, "../../../.env")
load_dotenv(env_path)


# === Data ===

risk_descriptions = {
    "R001": "Phishing attacks targeting employee emails",
    "R002": "Credential reuse due to poor password hygiene",
    "R003": "Insider misuse of admin access to exfiltrate sensitive data",
    "R004": "Delayed revocation of credentials after employee termination",
    "R005": "Privilege escalation vulnerability in legacy systems",
    "R006": "Weak audit trail for access to critical systems",
}

risk_pairs = [
    ("R002", "R001"),
    ("R004", "R003"),
    ("R003", "R006"),
    ("R005", "R004"),
]

# === Pydantic Model ===


class RiskRelationResult(BaseModel):
    interdependency_type: Literal[
        "Causal",
        "Correlated",
        "None",
    ]
    direction: Optional[Literal["A → B", "B → A", "None", "Both"]] = None
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


def summarize_risk(risk_data):
    llm = get_llm(model_name="gpt-4.1-mini")
    risk_name = risk_data["risk"]
    risk_desc = risk_data["risk_desc"]
    rootcause = risk_data["rootcause"]
    process = risk_data["process"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a risk analysis assistant reviewing pairs of risk statements. then summarize the risk to make it more readable. and concise. and make output in English.",
            ),
            (
                "human",
                f"Risk: {risk_name} \n Risk Description: {risk_desc} \n Root Cause: {rootcause} \n Process: {process} \n Output in English.",
            ),
        ]
    )
    structured_llm = llm.with_structured_output(RiskSummary)
    chain = prompt | structured_llm
    result = chain.invoke(risk_data)
    result = result.model_dump()
    result["risk"] = risk_name
    return result


def analyze_pair(risk_a, risk_b, analyze_model_name):
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
- A → B (Risk A leads to Risk B)
- B → A (Risk B leads to Risk A)
- Both (Bidirectional causality)
- None (For non-directional types)

Direction is required for Causal types.
For other types (Correlated, None), direction should be "None".

Provide the analysis in the specified JSON format.""",
                ),
                (
                    "human",
                    f"""Analyze the interdependency between the following two risks:
Risk A: {risk_a['risk']} \n Risk Description: {risk_a['risk_desc']} \n Root Cause: {risk_a['rootcause']} \n Process: {risk_a['process']}
Risk B: {risk_b['risk']} \n Risk Description: {risk_b['risk_desc']} \n Root Cause: {risk_b['rootcause']} \n Process: {risk_b['process']}""",
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
    # risk_data_path = "/Users/ford/Documents/coding_trae/cro_rmi_improvement_feature/src/cro_rmi_improvement_feature/plot_risk_related/result/250528-company_risk_data.json"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    risk_data_path = (
        f"{dir_path}/result/merge-company_risk_data_with_embedding-10percent_edges.pkl"
    )
    if risk_data_path.endswith(".pkl"):
        import pickle

        with open(risk_data_path, "rb") as f:
            risk_data = pickle.load(f)
    else:
        with open(risk_data_path, "r") as f:
            risk_data = json.load(f)
    print(f"{len(risk_data)=}")
    for node_edge in risk_data:
        print(f"{node_edge['data'].keys()=}")
    # random select 5 risk data)
    edge_data_list = []
    for risk in risk_data:
        # print(risk)  # pretty print
        source = risk["data"].get("source", None)
        if source == None:
            risk_id = risk["data"]["id"]
            continue
        company = source.split("_")[1]
        if company != "PCG":
            continue
        print(f"{company=}")

        edge_data_list.append(risk["data"])

    analyze_model_name_list = [
        # "gpt-4o-mini",
        "gpt-4.1-mini",
        "o3-mini",
    ]
    # selected_edge_data_list = edge_data_list
    # label_postfix = ""
    # 20% of total edge data
    selected_edge_data_list = random.sample(
        edge_data_list, int(len(edge_data_list) * 0.2)
    )
    label_postfix = "-label"

    print(len(selected_edge_data_list))
    print(selected_edge_data_list[0])

    all_results = []
    for analyze_model_name in analyze_model_name_list:
        print(f"\n===== Running analysis for model: {analyze_model_name} =====\n")
        result_list = []
        with get_openai_callback() as cb_summary:
            unique_risk_data = []
            for selected_edge_data in selected_edge_data_list:
                source_id = selected_edge_data["source"]
                risk_a = selected_edge_data["target_risk_data"]
                risk_b = selected_edge_data["source_risk_data"]
                if risk_a not in unique_risk_data:
                    unique_risk_data.append(risk_a)
                if risk_b not in unique_risk_data:
                    unique_risk_data.append(risk_b)
            for risk_data in unique_risk_data:
                risk_data_summary = summarize_risk(risk_data)
        all_cb_analyze = []
        for selected_edge_data in selected_edge_data_list:
            source_id = selected_edge_data["source"]
            risk_a = selected_edge_data["target_risk_data"]
            risk_b = selected_edge_data["source_risk_data"]
            for risk_a, risk_b, direction_risk in [
                (risk_a, risk_b, "a->b"),
                (risk_b, risk_a, "b->a"),
            ]:
                target_risk_data = risk_a
                source_risk_data = risk_b
                target_risk_data_summary = summarize_risk(target_risk_data)
                source_risk_data_summary = summarize_risk(source_risk_data)
                with get_openai_callback() as cb_analyze:
                    risk_relation_result = analyze_pair(
                        source_risk_data_summary,
                        target_risk_data_summary,
                        analyze_model_name,
                    )
                all_cb_analyze.append(cb_analyze)

                try:
                    target_risk_data_cell = f"risk_name: {target_risk_data_summary['risk']} \n risk_desc: {target_risk_data_summary['risk_desc']} \n rootcause: {target_risk_data_summary['rootcause']} \n process: {target_risk_data_summary['process']}"
                    source_risk_data_cell = f"risk_name: {source_risk_data_summary['risk']} \n risk_desc: {source_risk_data_summary['risk_desc']} \n rootcause: {source_risk_data_summary['rootcause']} \n process: {source_risk_data_summary['process']}"
                    relation_result = {
                        "direction_risk": direction_risk,
                        "analyze_model_name": analyze_model_name,
                        "target_risk": target_risk_data_summary["risk"],
                        "target_risk_data": target_risk_data_cell,
                        "source_risk": source_risk_data_summary["risk"],
                        "source_risk_data": source_risk_data_cell,
                        "interdependency_type": risk_relation_result[
                            "interdependency_type"
                        ],
                        "direction": risk_relation_result["direction"],
                        "rationale": risk_relation_result["rationale"],
                        "confidence": risk_relation_result["confidence"],
                    }
                except Exception as e:
                    print(f"Error: {risk_relation_result.keys()}")
                    print(f"Error: {risk_relation_result}")
                    raise e
                result_list.append(relation_result)
        print(f"total analyze cost: {sum(cb.total_cost for cb in all_cb_analyze)}")
        print(
            f"total analyze cost thai: {sum(cb.total_cost for cb in all_cb_analyze) * 35}"
        )
        print(f"total analyze tokens: {sum(cb.total_tokens for cb in all_cb_analyze)}")
        print(
            f"total analyze prompt tokens: {sum(cb.prompt_tokens for cb in all_cb_analyze)}"
        )
        print(
            f"total analyze completion tokens: {sum(cb.completion_tokens for cb in all_cb_analyze)}"
        )
        print(f"total summary cost: {cb_summary.total_cost}")
        print(f"total summary cost thai: {cb_summary.total_cost * 35}")
        print(f"total summary tokens: {cb_summary.total_tokens}")
        print(f"total summary prompt tokens: {cb_summary.prompt_tokens}")
        print(f"total summary completion tokens: {cb_summary.completion_tokens}")
        count_same_interdependency_type = 0
        count_same_direction = 0
        count_same_both = 0
        for i in range(0, len(result_list), 2):
            interdependency_type_a_b = result_list[i]["interdependency_type"]
            interdependency_type_b_a = result_list[i + 1]["interdependency_type"]
            direction_a_b = result_list[i]["direction"]
            direction_b_a = result_list[i + 1]["direction"]
            final_interdependency_type, final_direction = final_relationship(
                interdependency_type_a_b,
                interdependency_type_b_a,
                direction_a_b,
                direction_b_a,
            )
            result_list[i]["final_interdependency_type"] = final_interdependency_type
            result_list[i]["final_direction"] = final_direction
            result_list[i + 1][
                "final_interdependency_type"
            ] = final_interdependency_type
            result_list[i + 1]["final_direction"] = final_direction
            count_possible_missing_direction_relation = None
            if (final_interdependency_type in ["Causal"]) and (
                interdependency_type_a_b not in ["Causal"]
            ):
                count_possible_missing_direction_relation = True
            if interdependency_type_a_b in [
                "Causal",
            ]:
                count_possible_missing_direction_relation = False
            result_list[i][
                "count_possible_missing_direction_relation"
            ] = count_possible_missing_direction_relation
            result_list[i + 1][
                "count_possible_missing_direction_relation"
            ] = count_possible_missing_direction_relation
            if interdependency_type_a_b != interdependency_type_b_a:
                print(f"{interdependency_type_a_b=}")
                print(f"{interdependency_type_b_a=}")
                print()
            if (
                result_list[i]["interdependency_type"]
                == result_list[i + 1]["interdependency_type"]
            ):
                count_same_interdependency_type += 1
            if result_list[i]["direction"] == result_list[i + 1]["direction"]:
                count_same_direction += 1
            if (
                result_list[i]["interdependency_type"]
                == result_list[i + 1]["interdependency_type"]
                and result_list[i]["direction"] == result_list[i + 1]["direction"]
            ):
                count_same_both += 1
        print(f"total pairs: {len(result_list)/2}")
        print(f"{count_same_interdependency_type=}")
        print(f"{count_same_direction=}")
        print(f"{count_same_both=}")
        all_results.extend(result_list)
    # After all models, save to one Excel file
    df = pd.DataFrame(all_results)
    df.to_excel(
        f"{dir_path}/risk_relationship_classifier_result_all_models{label_postfix}.xlsx",
        index=False,
    )
