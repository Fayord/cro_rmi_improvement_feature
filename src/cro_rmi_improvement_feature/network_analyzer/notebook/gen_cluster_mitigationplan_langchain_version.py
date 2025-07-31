import re
import pickle
import networkx as nx
import pandas as pd
import os
import numpy as np
from networkx.classes.reportviews import InDegreeView, OutDegreeView
from networkx.algorithms.community import louvain_communities
import igraph as ig
import leidenalg

# langchain
# set up cache
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# set up cache
dir_path = os.path.dirname(os.path.realpath(__file__))
set_llm_cache(SQLiteCache(database_path=f"{dir_path}/.gen_mitigation.db"))
# from networkx.algorithms.community import louvain_communities, leiden_communities

# CONFIGS
# LLM_VENDOR = 'openai' # openai
# MODEL = "gpt-4.1" # "gpt-4o"
# N_TOP = 3


# ========================================== NETWORK ANALYSIS ==========================================
# === STEP 4: GENERATE LLM PROMPTS ===
def generate_prompt(cluster, G, nodes):
    prompt = "You are a risk analyst tasked with recommending systemic controls.\n"
    prompt += "Below is a cluster of interdependent risks. Suggest a minimal, systemic set of responses that reduce overall exposure:\n\n"

    prompt += "Risks:\n"
    for node in cluster:
        d = G.nodes[node]
        # prompt += f"- {node}: {d['description']} (level: {d['level']}, impact: {d['impact']})\n"
        # nodes["data"]["label"]

        # label = next((item['data']['label'] for item in data if item['data']['id'] == 'risk_PCG_0'), None)

        label = next(
            (item["data"]["label"] for item in nodes if item["data"]["id"] == node),
            None,
        )
        prompt += f"- {node}: {label}\n"
        # print(label)
        # print(node)

    prompt += "\nDependencies:\n"
    edges = G.subgraph(cluster).edges(data=True)
    for u, v, attr in edges:
        # prompt += f"- {u} ↔ {v} ({attr['type']})\n"
        # prompt += f"- {u} ↔ {v} \n"
        prompt += f"- {u} -> {v} \n"

    prompt += "\nPlease suggest 1–3 bundled or coordinated controls or mitigation strategies that address this cluster holistically.\n"

    # Replace all occurrences of risk_PCG_XX with risk_XX
    cleaned_prompt = re.sub(r"risk_[^_]+_(\d+)", r"risk_\1", prompt)
    return cleaned_prompt
    # return prompt


def generate_prompt_nointro(cluster, G, nodes):
    prompt = "Risks:\n"
    for node in cluster:
        d = G.nodes[node]
        # prompt += f"- {node}: {d['description']} (level: {d['level']}, impact: {d['impact']})\n"
        # nodes["data"]["label"]

        # label = next((item['data']['label'] for item in data if item['data']['id'] == 'risk_PCG_0'), None)

        label = next(
            (item["data"]["label"] for item in nodes if item["data"]["id"] == node),
            None,
        )
        prompt += f"- {node}: {label}\n"
        # print(label)
        # print(node)

    prompt += "\nDependencies:\n"
    edges = G.subgraph(cluster).edges(data=True)
    for u, v, attr in edges:
        # prompt += f"- {u} ↔ {v} ({attr['type']})\n"
        # prompt += f"- {u} ↔ {v} \n"
        prompt += f"- {u} -> {v} \n"

    # prompt += "\nPlease suggest 1–3 bundled or coordinated controls or mitigation strategies that address this cluster holistically.\n"

    # Replace all occurrences of risk_PCG_XX with risk_XX
    cleaned_prompt = re.sub(r"risk_[^_]+_(\d+)", r"risk_\1", prompt)
    return cleaned_prompt


# set up backends for leiden
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.leiden.leiden_communities.html#networkx.algorithms.community.leiden.leiden_communities
# 'nx-parallel' has no support for leiden, need to use 'cugraph'


def filter_non_arrow_edges(edges):
    return [edge for edge in edges if edge["data"]["arrow_weight"] != "none"]


def get_number_edges_to_show(total_nodes):
    return 2 * total_nodes


def filter_non_arrow_edges2(edges, slider_value):
    new_edges = []
    for edge in edges:
        if edge["interdependency_type"] == "Causal":
            # if edge["direction"] is not None:
            # if edge["direction"] != "none":
            if edge["cosine_similarity"] >= slider_value:
                new_edges.append(edge)
    return new_edges


# multiple companies in one pickle file
def find_graph_properties_newpickle(
    data_path, company_name, CLUSTER_METHOD="leiden"
):  # sink/source/central nodes; clusters
    data_all = pickle.load(open(data_path, "rb"))

    # data = data_all['company_graph_datas'][company_name]

    all_data_list = []
    # for i, company_name in enumerate(data_all['company_graph_datas'].keys()):
    G = nx.DiGraph()

    data = data_all["company_graph_datas"][company_name]
    # data = pickle.load(open(data_path, "rb"))

    nodes = data["nodes"]
    edges = data["edges"]
    line_weights = [edge["cosine_similarity"] for edge in edges]
    num_edges_to_show = get_number_edges_to_show(len(nodes))
    sorted_weights = sorted(line_weights, reverse=True)
    # The threshold is the weight of the (num_edges_to_show)-th edge (0-indexed)
    slider_value = sorted_weights[num_edges_to_show - 1]
    filter_edges = filter_non_arrow_edges2(edges, slider_value)
    for edge in filter_edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["distance"],
        )
    in_degree_centrality_dict = nx.in_degree_centrality(G)
    out_degree_centrality_dict = nx.out_degree_centrality(G)
    betweenness_dict_weight = nx.betweenness_centrality(G, weight="weight")
    betweenness_dict_non_weight = nx.betweenness_centrality(G)

    # create list of data so I can convert to dataframe later
    for node in nodes:
        # in_deg = G.in_degree(node["data"]["id"])
        # in_deg = 0 if in_deg in ([], (), None) else in_deg
        row_data = {
            # "company": company,
            "risk_id": node["data"]["id"],
            "risk_name": node["data"]["label"],
            "risk_level": node["data"]["risk_level"],
            # "in_degree": in_deg,
            "in_degree": G.in_degree(node["data"]["id"]),
            "out_degree": G.out_degree(node["data"]["id"]),
            "in_degree_centrality": in_degree_centrality_dict.get(
                node["data"]["id"], None
            ),
            "out_degree_centrality": out_degree_centrality_dict.get(
                node["data"]["id"], None
            ),
            "betweenness_centrality_weight": betweenness_dict_weight.get(
                node["data"]["id"], None
            ),
            "betweenness_centrality_non_weight": betweenness_dict_non_weight.get(
                node["data"]["id"], None
            ),
        }
        # if not row_data['out_degree'].is_integer():
        #     print('should be 0')
        all_data_list.append(row_data)

    # === STEP 3: CLUSTER RISKS ===
    # louvain
    if CLUSTER_METHOD == "louvain":
        clusters = louvain_communities(G)

    # leiden
    if CLUSTER_METHOD == "leiden":
        # clusters = leiden_communities(G, backend="cugraph") # this method needs cugraph backend
        # Step 3.2: Convert NetworkX to igraph
        # G_ig = ig.Graph.TupleList(G.edges(), directed=False)
        G_ig = ig.Graph.TupleList(G.edges(), directed=True, vertex_name_attr="name")

        # Step 3.3: Run Leiden algorithm
        # partition = leidenalg.find_partition(G_ig, leidenalg.CPMVertexPartition) # doesn't seem to work with directed graph
        partition = leidenalg.find_partition(G_ig, leidenalg.RBERVertexPartition)
        # partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition) # for non-directed

        # Step 3.4: Extract communities (as lists of original node names)
        clusters = [
            [G_ig.vs[node]["name"] for node in community] for community in partition
        ]

        # print("Leiden communities:", clusters)

        # # # === STEP 5: OUTPUT PROMPTS ===
        # for i, cluster in enumerate(clusters, 1):
        #     print(f"\n--- Cluster {i} Prompt ---\n")
        #     print(generate_prompt(cluster, G, nodes))

    all_data_df = pd.DataFrame(all_data_list)

    all_data_df["in_degree"] = all_data_df["in_degree"].apply(
        lambda x: np.nan if isinstance(x, InDegreeView) else int(x)
    )
    all_data_df["out_degree"] = all_data_df["out_degree"].apply(
        lambda x: np.nan if isinstance(x, OutDegreeView) else int(x)
    )
    # all_data_df.head()
    return [all_data_df, clusters, G, nodes]


def find_graph_properties(
    data_path_dict, CLUSTER_METHOD="leiden"
):  # sink/source/central nodes; clusters
    all_data_list = []
    for company, data_path in data_path_dict.items():
        G = nx.DiGraph()
        data = pickle.load(open(data_path, "rb"))

        nodes = data[0]
        edges = data[1]
        filter_edges = filter_non_arrow_edges(edges)
        for edge in filter_edges:
            G.add_edge(
                edge["data"]["source"],
                edge["data"]["target"],
                weight=edge["data"]["raw_weight"],
            )
        in_degree_centrality_dict = nx.in_degree_centrality(G)
        out_degree_centrality_dict = nx.out_degree_centrality(G)
        betweenness_dict_weight = nx.betweenness_centrality(G, weight="weight")
        betweenness_dict_non_weight = nx.betweenness_centrality(G)

        # create list of data so I can convert to dataframe later
        for node in nodes:
            # in_deg = G.in_degree(node["data"]["id"])
            # in_deg = 0 if in_deg in ([], (), None) else in_deg
            row_data = {
                "company": company,
                "risk_id": node["data"]["id"],
                "risk_name": node["data"]["label"],
                "risk_level": node["data"]["risk_level"],
                # "in_degree": in_deg,
                "in_degree": G.in_degree(node["data"]["id"]),
                "out_degree": G.out_degree(node["data"]["id"]),
                "in_degree_centrality": in_degree_centrality_dict.get(
                    node["data"]["id"], None
                ),
                "out_degree_centrality": out_degree_centrality_dict.get(
                    node["data"]["id"], None
                ),
                "betweenness_centrality_weight": betweenness_dict_weight.get(
                    node["data"]["id"], None
                ),
                "betweenness_centrality_non_weight": betweenness_dict_non_weight.get(
                    node["data"]["id"], None
                ),
            }
            # if not row_data['out_degree'].is_integer():
            #     print('should be 0')
            all_data_list.append(row_data)

        # === STEP 3: CLUSTER RISKS ===
        # louvain
        if CLUSTER_METHOD == "louvain":
            clusters = louvain_communities(G)

        # leiden
        if CLUSTER_METHOD == "leiden":
            # clusters = leiden_communities(G, backend="cugraph") # this method needs cugraph backend
            # Step 3.2: Convert NetworkX to igraph
            # G_ig = ig.Graph.TupleList(G.edges(), directed=False)
            G_ig = ig.Graph.TupleList(G.edges(), directed=True, vertex_name_attr="name")

            # Step 3.3: Run Leiden algorithm
            # partition = leidenalg.find_partition(G_ig, leidenalg.CPMVertexPartition) # doesn't seem to work with directed graph
            partition = leidenalg.find_partition(G_ig, leidenalg.RBERVertexPartition)
            # partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition) # for non-directed

            # Step 3.4: Extract communities (as lists of original node names)
            clusters = [
                [G_ig.vs[node]["name"] for node in community] for community in partition
            ]

            # print("Leiden communities:", clusters)

        # # # === STEP 5: OUTPUT PROMPTS ===
        # for i, cluster in enumerate(clusters, 1):
        #     print(f"\n--- Cluster {i} Prompt ---\n")
        #     print(generate_prompt(cluster, G, nodes))

    all_data_df = pd.DataFrame(all_data_list)

    all_data_df["in_degree"] = all_data_df["in_degree"].apply(
        lambda x: np.nan if isinstance(x, InDegreeView) else int(x)
    )
    all_data_df["out_degree"] = all_data_df["out_degree"].apply(
        lambda x: np.nan if isinstance(x, OutDegreeView) else int(x)
    )
    # all_data_df.head()
    return [all_data_df, clusters, G, nodes]


# all_data_df, clusters, G, nodes = find_graph_properties(data_path_dict)


def top_n_with_strict_limit(df, column, n=5):
    # Get the top N unique values
    top_unique_values = (
        df[column].drop_duplicates().sort_values(ascending=False).tolist()
    )

    # Limit to top N values only
    limited_top_values = top_unique_values[:n]

    # If there are more than N rows due to ties, trim down
    result = df[df[column].isin(limited_top_values)]

    # Check if result is too long (due to tie at N-th value)
    if len(result) > n:
        # Find the N-th value
        nth_value = limited_top_values[-1]
        # Drop all rows with N-th value
        result = result[result[column] != nth_value]

    # return result
    return result.sort_values(by=column, ascending=False)  # return sorted list


def top_n_with_row_limit(df, column, n=5):
    df_sorted = df.sort_values(by=column, ascending=False)
    counts = df_sorted[column].value_counts().sort_index(ascending=False)

    total = 0
    allowed_values = []

    for value, count in counts.items():
        if total + count <= n:
            allowed_values.append(value)
            total += count
        else:
            break  # skip this value and stop

    result = df_sorted[df_sorted[column].isin(allowed_values)]
    return result


def find_sublist_containing(target, list_of_lists):
    for sublist in list_of_lists:
        if target in sublist:
            return sublist
    return None  # or [] if you prefer


def find_sublists_with_any(targets, list_of_lists):
    return [sublist for sublist in list_of_lists if any(t in sublist for t in targets)]


# example 1
# estimate_cost.total_cost_THB = 0 # initialization
# estimate_cost(res_usage = usage_count_list,type='multi',model='gpt-4o-mini')

# example 2, use default value of 'single' run and '4o' model
# estimate_cost(res_usage = response.usage)
# estimate_cost(response.usage)

# ================================================ PYDANTIC STUFF ================================================
from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional

# References:
# 2004 COSO ERM - Integrated Framework - Application Techniques.pdf, p.43 (51)
# _COSO Enterprise Risk Management - Integrating with Strategy and Performance.pdf, p.82 (88)


class control_action_with_related_risk(BaseModel):
    action: str = Field(
        ..., description="Action, policy, or system to be put in place."
    )
    risks_covered: str = Field(
        ...,
        description="List the risk codes or descriptions impacted directly or indirectly by this action",
    )
    duration: str = Field(
        ...,
        description="Short-term (0–6 months), medium (6–12 months), or long-term (1 year+). Explain the rationale.",
    )  # TODO: change to literal?


class risk_addressed_and_target(BaseModel):
    risk_addressed: str = Field(
        ...,
        description="risk code or descriptions impacted directly or indirectly by this plan",
    )
    target_risk_likelihood: str = Field(
        ...,
        description="target risk likelihood after the plan completion: Rare, Unlikely, Moderate, High, or Almost Certain",
    )
    target_risk_impact: str = Field(
        ...,
        description="target risk impact after the plan completion: Very Low, Minor, Medium, Major, or Critical",
    )


class DetailedActionPlanCluster(BaseModel):  # TODO: modify for control_cluster plan
    cluster_plan_name: str = Field(
        ...,
        description="A clear and concise title that reflects the nature or focus of the response (e.g., “Agile Innovation Pipeline” or “Talent Resilience Program”)",
    )
    objective: str = Field(
        ...,
        description="A brief explanation of the systemic challenge the plan addresses and why this approach is selected (e.g., “To reduce exposure to rapidly evolving competition and delayed innovation by enhancing adaptive capability.”)",
    )
    priority_level: str = Field(
        ..., description="Plan's priority level: Low, Medium, or High"
    )
    key_impact: str = Field(
        ...,
        description="Describe how the intervention diffuses through the network of dependencies to reduce overall exposure.",
    )
    # risks_addressed: List[str] = Field(...,description="List the risk codes or descriptions impacted directly or indirectly by this plan")
    risks_addressed: List[risk_addressed_and_target] = Field(
        ...,
        description="List the risk codes or descriptions impacted directly or indirectly by this plan",
    )
    # control_actions: List[str] = Field(...,description="A set of coordinated actions, policies, or systems to be put in place. Should include both technical/process and organizational levers.")
    control_actions: List[control_action_with_related_risk] = Field(
        ...,
        description="A set of coordinated actions, policies, or systems to be put in place. Should include both technical/process and organizational levers.",
    )
    # cascading_effect: str = Field(...,description=" Explain how addressing direct risks reduces dependent risks")
    cascading_effect: str = Field(
        ...,
        description="Explain clearly how all risks listed in `risks_addressed` are causally or indirectly connected through the dependency graph. Show the logical or temporal sequence of how controlling one or more risks leads to the reduction or prevention of the others.",
    )
    metrics: List[str] = Field(
        ...,
        description="Quantitative or qualitative signals that track effectiveness of the controls.",
    )
    responsible_party: str = Field(
        ...,
        description="Who is accountable for the implementation and monitoring (e.g., Innovation Team, HR, Product Operations)",
    )
    timeline: str = Field(
        ...,
        description="Short-term (0–6 months), medium (6–12 months), or long-term (1 year+). Can be phased.",
    )  # TODO: change to literal? Literal["Short-term", "Medium-term", "Long-term"]
    expected_cost: str = Field(..., description="Plan cost in THB, $, and CNY")
    expected_financial_benefit: str = Field(
        ...,
        description="Financial benefit when implemented the plan in THB, $, and CNY",
    )
    expected_nonfinancial_benefit: str = Field(
        ..., description="Non-financial benefits when implemented the plan"
    )
    # estimated_impact_reduction: str = Field(...,description="Low, Moderate, High, or Very-High. Provide supporting reasons.")
    # estimated_likelihood_reduction: str = Field(...,description="Low, Moderate, High, or Very-High. Provide supporting reasons.")
    support_documents: str = Field(..., description="possible supporting documents")
    # kpis: str
    cost_benefit_analysis: str
    resources_required: str
    # effectiveness_in_scenarios: List[str]  # Describe effectiveness for each scenario.


class RiskMitigationActionClusterPlan(
    BaseModel
):  # TODO: modify for control_cluster plan
    detailed_action_plan: List[DetailedActionPlanCluster]


# ================================================ MITIGATION PLAN GENERATION ================================================
# TODO: add company's user-input data

from dotenv import load_dotenv
import os
from typing import List, Tuple

load_dotenv("../../.env")


import json
from textwrap import dedent

# from openai import OpenAI # commented out for langchain


# if LLM_VENDOR == 'openai':
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#     # MODEL = "gpt-4o-2024-08-06"
#     # MODEL = "gpt-4o"
# elif LLM_VENDOR == 'lmstudio':
#     client = OpenAI(
#         base_url="http://localhost:1234/v1",
#         api_key="lm-studio"
#     )
#     # MODEL = "lmstudio-community/qwen2.5-7b-instruct-1m"


# system_prompt = '''
#     1. You are a highly skilled risk analyst with expertise in identifying systemic controls for clusters of interdependent risks. Your goal is to suggest minimal, but effective systemic sets of responses that reduce overall risk exposure.
#     2. Answer in English
# '''

# TODO 250721:
# Y add support for dynamic list of risks (see generate_prompt in network_analysis.ipynb)
# - add support for different types of optimization:
#   a. (best) minimal, systemic set of responses
#   b. min # of control with the most coverage
#   c. min cost
#   d. min time
#   e. single control with the most coverage
# Y change to one-way arrow?
# Y add more attributes to plan generation pydantic (DetailedActionPlanCluster) to match with Figma


def get_response_control_cluster(cluster_risks, model="gpt-4o-mini"):
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # commented out for langchain
    # MODEL = "gpt-4.1" # "gpt-4o"
    # response = client.beta.chat.completions.parse(
    #     model=MODEL,
    #     temperature=0.7, # 0.3 - more deterministic,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": dedent('''
    # 1. You are a highly skilled risk analyst with expertise in identifying systemic controls for clusters of interdependent risks. Your goal is to suggest minimal, but effective systemic sets of responses that reduce overall risk exposure.
    # 2. Answer in English
    # ''')
    #         },
    #         {
    #             "role": "user",
    #             "content": f'''You are a risk analyst tasked with recommending systemic controls for a cluster of interdependent risks.
    #             Your goal is to propose 3 to 5 **systemic mitigation plans** that address the risks holistically.
    # Below is a cluster of interdependent risks. Suggest options for a minimal, systemic set of responses that reduce overall exposure:
    #
    # {cluster_risks}
    # '''
    #         }
    #     ],
    #     response_format=RiskMitigationActionClusterPlan
    # )

    llm = ChatOpenAI(model=model, temperature=0.7)
    structured_llm = llm.with_structured_output(RiskMitigationActionClusterPlan)

    # system_prompt = f"""
    # 1. You are a highly skilled risk analyst with expertise in identifying systemic controls for clusters of interdependent risks. Your goal is to suggest minimal, but effective systemic sets of responses that reduce overall risk exposure.
    # 2. Answer in English
    # """

    messages = [
        SystemMessage(
            dedent(
                """
            1. You are a highly skilled risk analyst with expertise in identifying systemic controls for clusters of interdependent risks. Your goal is to suggest minimal, but effective systemic sets of responses that reduce overall risk exposure.
            2. Answer in English
        """
            )
        ),
        HumanMessage(
            content=f"""You are a risk analyst tasked with recommending systemic controls for a cluster of interdependent risks.
            Your goal is to propose 3 to 5 **systemic mitigation plans** that address the risks holistically.
        Below is a cluster of interdependent risks. Suggest options for a minimal, systemic set of responses that reduce overall exposure:

        {cluster_risks}
        """
        ),
    ]
    response = structured_llm.invoke(messages)

    return response


# response.choices[0].message.tool_calls
# print(response['choices'][0]['message']['content'])


def get_response_test(cluster_risks, model="gpt-4o-mini"):
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # MODEL = "gpt-4.1" # "gpt-4o"
    # response = client.beta.chat.completions.parse(
    #     model=MODEL,
    #     temperature=0.7, # 0.3 - more deterministic,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You will greet the other side."
    #         },
    #         {
    #             "role": "user",
    #             "content": "How are you"
    #         }
    #     ],
    #     # response_format=RiskMitigationActionClusterPlan
    # )
    llm = ChatOpenAI(model=model, temperature=0.7)
    structured_llm = llm.with_structured_output(RiskMitigationActionClusterPlan)
    messages = [
        SystemMessage("You will greet the other side."),
        HumanMessage(content="How are you"),
    ]
    response = structured_llm.invoke(messages)

    return response


# ================= MAIN/ =================


# ================= /MAIN =================
