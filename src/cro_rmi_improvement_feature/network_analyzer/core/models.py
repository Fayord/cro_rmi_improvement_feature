"""
Pydantic schemas for data models in the Risk Network Visualization System.

This module defines the data structures used throughout the application,
including risk nodes, edges, and graph representations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class Process(BaseModel):
    """Schema for process data."""

    id: str = Field(..., description="Unique identifier for the process")
    name: str = Field(..., description="Name of the process")
    description: str = Field(..., description="Description of the process")

    class Config:
        schema_extra = {
            "example": {
                "id": "1",
                "name": "Procurement",
                "description": "Vendor selection and management process",
            }
        }


class RootCause(BaseModel):
    """Schema for root cause data."""

    id: str = Field(..., description="Unique identifier for the root cause")
    name: str = Field(..., description="Name of the root cause")
    description: str = Field(..., description="Description of the root cause")

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Single source dependency",
                "description": "Over-reliance on single supplier",
            }
        }


class RiskData(BaseModel):
    id: str = Field(..., description="The id of the risk node.")
    label: str = Field(..., description="Text to display for the risk node.")
    risk: str = Field(..., description="The name of the risk.")
    risk_id: str = Field(..., description="The id of the risk.")
    risk_cat: str = Field(..., description="The category of the risk.")
    risk_desc: Union[List[str], str] = Field(
        ..., description="The description of the risk."
    )
    risk_score: int = Field(..., description="The score of the risk.")
    risk_impact: int = Field(..., description="The impact of the risk.")
    risk_likelihood: int = Field(..., description="The likelihood of the risk.")
    risk_level: int = Field(..., description="The level of the risk.")
    process: Union[List[Process], List[str], str, None] = Field(
        ..., description="The process of the risk."
    )
    rootcause: Union[List[RootCause], List[str], str, None] = Field(
        ..., description="The root cause of the risk."
    )
    process_summary: Optional[str] = Field(
        None, description="The summary of the process."
    )
    rootcause_summary: Optional[str] = Field(
        None, description="The summary of the root cause."
    )
    risk_desc_summary: Optional[str] = Field(
        ..., description="The summary of the risk description."
    )


class RiskDataNoEmbedding(RiskData):
    """
    A version of RiskData without the embedding field, for cases where only data is needed.
    """

    pass


class RiskDataWithEmbedding(BaseModel):
    data: RiskData = Field(..., description="The risk node data.")
    embedding: List[float] = Field(..., description="The embedding of the risk node.")


class RiskDataWithOutEmbedding(BaseModel):
    data: RiskDataNoEmbedding = Field(..., description="The risk node data.")


class RiskOverlayData(BaseModel):
    overlay_data: Dict[str, List[RiskDataWithEmbedding]] = Field(
        ..., description="A dictionary mapping risk name to list of risk data"
    )


class RiskOverlayDataNoEmbedding(BaseModel):
    overlay_data: Dict[str, List[RiskDataWithOutEmbedding]] = Field(
        ..., description="A dictionary mapping risk name to list of risk data"
    )


class EdgeData(BaseModel):
    source: str = Field(..., description="The source of the edge. risk_a_data.id")
    target: str = Field(..., description="The target of the edge. risk_b_data.id")
    interdependency_type: Optional[str] = Field(
        None, description="The interdependency type of the edge."
    )
    direction: Optional[str] = Field(None, description="The direction of the edge.")
    rationale: Optional[str] = Field(None, description="The rationale of the edge.")
    confidence: Optional[float] = Field(None, description="The confidence of the edge.")
    risk_a_data: RiskData = Field(..., description="The risk node data of risk a.")
    risk_b_data: RiskData = Field(..., description="The risk node data of risk b.")
    distance: float = Field(..., description="The distance between the two risks.")
    cosine_similarity: float = Field(
        ..., description="The cosine similarity of the edge."
    )
    high_priority: bool = Field(
        False,
        description="it is a primary connected edge to high risk",
    )
    similarity_rank: int = Field(
        ..., description="The similarity rank of the edge start from 0"
    )


class EdgeDataNoEmbedding(BaseModel):
    source: str = Field(..., description="The source of the edge. risk_a_data.id")
    target: str = Field(..., description="The target of the edge. risk_b_data.id")
    interdependency_type: Optional[str] = Field(
        None, description="The interdependency type of the edge."
    )
    direction: Optional[str] = Field(None, description="The direction of the edge.")
    rationale: Optional[str] = Field(None, description="The rationale of the edge.")
    confidence: Optional[float] = Field(None, description="The confidence of the edge.")
    risk_a_data: RiskDataNoEmbedding = Field(
        ..., description="The risk node data of risk a."
    )
    risk_b_data: RiskDataNoEmbedding = Field(
        ..., description="The risk node data of risk b."
    )
    distance: float = Field(..., description="The distance between the two risks.")
    cosine_similarity: float = Field(
        ..., description="The cosine similarity of the edge."
    )
    high_priority: bool = Field(
        False,
        description="it is a primary connected edge to high risk",
    )
    similarity_rank: int = Field(
        ..., description="The similarity rank of the edge start from 0"
    )


class CompanyGraphData(BaseModel):
    """
    Represents the graph data for a specific company and embedding type.
    This is a flexible structure; you can make it more specific
    (e.g., nodes: List[Node], edges: List[Edge]) if needed.
    """

    nodes: List[RiskDataWithEmbedding] = Field(
        ..., description="List of node objects, each with properties."
    )
    edges: List[EdgeData] = Field(
        ...,
        description="List of edge objects, each with source, target, and properties.",
    )
    number_of_displayed_edges: Optional[int] = Field(
        None, description="Number of edges to display in the graph."
    )
    risk_catalog_reference_id: Optional[str] = Field(
        None, description="Version of the risk catalog used for reference."
    )
    overlay_top_n_risks_catalog_id: Optional[str] = Field(
        None, description="Version of the top n risks catalog."
    )
    overlay_news_id_list: Optional[List[str]] = Field(
        None, description="List of news ids to overlay sub graph"
    )


class CompanyGraphDataNoEmbedding(BaseModel):
    """
    Represents the graph data for a specific company and embedding type, without embedding data.
    """

    nodes: List[RiskDataWithOutEmbedding] = Field(
        ..., description="List of node objects, each with properties."
    )
    edges: List[EdgeDataNoEmbedding] = Field(
        ...,
        description="List of edge objects, each with source, target, and properties.",
    )
    number_of_displayed_edges: Optional[int] = Field(
        None, description="Number of edges to display in the graph."
    )
    risk_catalog_reference_id: Optional[str] = Field(
        None, description="Version of the risk catalog used for reference."
    )
    overlay_top_n_risks_catalog_id: Optional[str] = Field(
        None, description="Version of the top n risks catalog."
    )
    overlay_news_id_list: Optional[List[str]] = Field(
        None, description="List of news ids to overlay sub graph"
    )


class GraphDataLibrary(BaseModel):
    """
    The main schema representing graph data organized by embedding type -> company -> graph data
    """

    company_graph_datas: Dict[str, CompanyGraphData] = Field(
        ...,
        description="A dictionary mapping company_graph_id str join by '|' (datestamp|company|embedding_source_type) to its CompanyGraphData.",
    )

    risk_overlay_datas: Dict[str, RiskOverlayData] = Field(
        ...,
        description="A dictionary mapping risk_overlay_id str join by '|' (overlay_type|timestamp|embedding_source_type or overlay_name) to its RiskOverlayData.",
    )
    risk_catalog_reference_datas: Dict[str, List[RiskDataWithEmbedding]] = Field(
        ...,
        description="A dictionary mapping timestamp to the risk data.",
    )


class GraphDataLibraryNoEmbedding(BaseModel):
    """
    The main schema representing graph data organized by embedding type -> company -> graph data, without embedding data.
    """

    company_graph_datas: Dict[str, CompanyGraphDataNoEmbedding] = Field(
        ...,
        description="A dictionary mapping company_graph_id str join by '|' (datestamp|company|embedding_source_type) to its CompanyGraphData.",
    )

    risk_overlay_datas: Dict[str, RiskOverlayDataNoEmbedding] = Field(
        ...,
        description="A dictionary mapping risk_overlay_id str join by '|' (overlay_type|timestamp|embedding_source_type or overlay_name) to its RiskOverlayData.",
    )
    risk_catalog_reference_datas: Dict[str, List[RiskDataWithOutEmbedding]] = Field(
        ...,
        description="A dictionary mapping timestamp to the risk data.",
    )
