"""
Pydantic schemas for the risk recommendation API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Process(BaseModel):
    """Schema for process data."""

    id: int = Field(..., description="Unique identifier for the process")
    name: str = Field(..., description="Name of the process")
    description: str = Field(..., description="Description of the process")

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Procurement",
                "description": "Vendor selection and management process",
            }
        }


class RootCause(BaseModel):
    """Schema for root cause data."""

    id: int = Field(..., description="Unique identifier for the root cause")
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


class RiskScore(BaseModel):
    """Schema for risk scoring data."""

    impact: int = Field(..., ge=1, le=5, description="Impact score (1 to 5)")
    likelihood: int = Field(..., ge=1, le=5, description="Likelihood score (1 to 5)")
    score: int = Field(..., ge=1, le=25, description="Overall risk score (1 to 25)")
    risk_level: int = Field(..., ge=1, le=4, description="Risk level (1, 2, 3, 4)")

    class Config:
        schema_extra = {
            "example": {"impact": 4, "likelihood": 3, "score": 12, "risk_level": 3}
        }


class ExistingRisk(BaseModel):
    """Schema for existing risk data."""

    risk_id: str = Field(..., description="Unique identifier for the risk")
    risk_name: str = Field(..., description="Name of the risk")
    risk_description: str = Field(..., description="Description of the risk")
    processes: List[Process] = Field(
        ..., description="List of processes associated with the risk"
    )
    root_causes: List[RootCause] = Field(
        ..., description="List of root causes for the risk"
    )
    risk_category: str = Field(..., description="Category of the risk")
    score: RiskScore = Field(..., description="Risk scoring information")

    class Config:
        schema_extra = {
            "example": {
                "risk_id": "risk_001",
                "risk_name": "Supply Chain Disruption",
                "risk_description": "Risk of disruption in supply chain operations",
                "processes": [
                    {
                        "id": 1,
                        "name": "Procurement",
                        "description": "Vendor selection and management process",
                    }
                ],
                "root_causes": [
                    {
                        "id": 1,
                        "name": "Single source dependency",
                        "description": "Over-reliance on single supplier",
                    }
                ],
                "risk_category": "operational",
                "score": {"impact": 4, "likelihood": 3, "score": 12, "risk_level": 3},
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
                        "processes": [
                            {
                                "id": 1,
                                "name": "Procurement",
                                "description": "Vendor selection and management process",
                            }
                        ],
                        "root_causes": [
                            {
                                "id": 1,
                                "name": "Single source dependency",
                                "description": "Over-reliance on single supplier",
                            }
                        ],
                        "risk_category": "operational",
                        "score": {
                            "impact": 4,
                            "likelihood": 3,
                            "score": 12,
                            "risk_level": 3,
                        },
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
