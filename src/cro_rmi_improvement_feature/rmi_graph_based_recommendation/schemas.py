"""
Pydantic schemas for the risk recommendation API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExistingRisk(BaseModel):
    """Schema for existing risk data."""

    risk_id: str = Field(..., description="Unique identifier for the risk")
    risk_name: str = Field(..., description="Name of the risk")
    risk_description: str = Field(..., description="Description of the risk")
    process: str = Field(..., description="Process associated with the risk")
    process_description: str = Field(..., description="Description of the process")
    root_cause: str = Field(..., description="Root cause of the risk")
    root_cause_description: str = Field(
        ..., description="Description of the root cause"
    )
    risk_category: str = Field(..., description="Category of the risk")

    class Config:
        schema_extra = {
            "example": {
                "risk_id": "risk_001",
                "risk_name": "Supply Chain Disruption",
                "risk_description": "Risk of disruption in supply chain operations",
                "process": "Procurement",
                "process_description": "Vendor selection and management process",
                "root_cause": "Single source dependency",
                "root_cause_description": "Over-reliance on single supplier",
                "risk_category": "operational",
            }
        }


class RiskRecommendationRequest(BaseModel):
    """Request schema for risk recommendations."""

    user_id: str = Field(..., description="Unique identifier for the user")

    existing_risks: List[ExistingRisk] = Field(
        ..., description="List of existing risks to consider"
    )

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "existing_risks": [
                    {
                        "risk_id": "risk_001",
                        "risk_name": "Supply Chain Disruption",
                        "risk_description": "Risk of disruption in supply chain operations",
                        "process": "Procurement",
                        "process_description": "Vendor selection and management process",
                        "root_cause": "Single source dependency",
                        "root_cause_description": "Over-reliance on single supplier",
                        "risk_category": "operational",
                    }
                ],
            }
        }


class RecommendedRisk(BaseModel):
    """Schema for recommended risk items."""

    risk_id: str = Field(..., description="Unique identifier for the risk")
    risk_name: str = Field(..., description="Name of the risk")

    class Config:
        schema_extra = {
            "example": {
                "risk_id": "risk_002",
                "risk_name": "Cybersecurity Breach",
            }
        }


class RiskRecommendationResponse(BaseModel):
    """Response schema for risk recommendations."""

    user_id: str = Field(..., description="User identifier")
    recommendations: List[RecommendedRisk] = Field(
        ..., description="List of recommended risks"
    )

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "recommendations": [
                    {
                        "risk_id": "risk_002",
                        "risk_name": "Cybersecurity Breach",
                    },
                    {
                        "risk_id": "risk_003",
                        "risk_name": "Regulatory Compliance Failure",
                    },
                ],
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request parameters",
                "detail": "user_id is required",
                "status_code": 400,
            }
        }
