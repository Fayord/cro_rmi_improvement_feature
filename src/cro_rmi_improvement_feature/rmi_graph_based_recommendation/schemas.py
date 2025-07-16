"""
Pydantic schemas for the risk recommendation API.
"""

from typing import List, Optional, Literal
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


class FileSchema(BaseModel):
    """Schema for an uploaded file."""

    name: str
    size: str  # e.g., "56.34 KB"
    status: str  # e.g., "X" (for delete) or "uploaded"


class TaskSchema(BaseModel):
    """Schema for an individual task."""

    pre_defined_task: str
    weighted_activity_percent: Optional[str] = Field(
        None,
        description="Weighted activity percent (optional because it can be default to 100/n_tasks but not be accurate)",
    )
    status: Literal["Not Started", "In Progress", "Completed"] = Field(
        ...,
        description="Status of the task (e.g., 'Not Started', 'In Progress', 'Completed')",
    )
    task_owner_placeholder: Optional[str] = Field(
        None,
        description="Placeholder for the task owner (optional because it need to list of candidate task owner from the user)",
    )
    task_owner_department: Optional[str] = Field(
        None,
        description="Department of the task owner (optional because it need to list of candidate task owner from the user)",
    )
    task_owner_email: Optional[str] = Field(
        None,
        description="Email of the task owner (optional because it need to list of candidate task owner from the user)",
    )
    start_date: str = Field(
        ...,
        description="Start date of the task (e.g., 'Select date' or actual date)",
    )
    end_date: str = Field(
        ...,
        description="End date of the task (e.g., 'Select date' or actual date)",
    )


class PlanDetailSchema(BaseModel):
    """Schema for the 'Plan Detail' section."""

    mitigation_plan_name: str
    mitigation_plan_objective: str
    linked_risks: List[str]  # e.g., ["Regulatory Compliance Failure"]
    priority_level: Literal["Low", "Medium", "High"]
    plan_owner: Optional[str] = Field(
        None,
        description="Name of the plan owner (optional because a list of candidate plan owners is needed from the user)",
    )
    plan_owner_email: Optional[str] = Field(
        None,
        description="Email of the plan owner (optional because a list of candidate plan owner emails is needed from the user)",
    )


class TimelineBudgetSchema(BaseModel):
    """Schema for the 'Timeline & Budget' section."""

    start_date: Optional[str] = Field(
        None,
        description="Start date of the project (optional because it can do current_data +estimate project_duration)",
    )
    end_date: Optional[str] = Field(
        None,
        description="End date of the project (optional because it can do current_data +estimate project_duration)",
    )

    expected_cost: Optional[float] = Field(
        None,
        description="Expected cost of the project (optional because it can make a guess but unlikly to be accurate without related data)",
    )
    expected_financial_benefit: Optional[float] = Field(
        None,
        description="Expected financial benefit of the project (optional because it can make a guess but unlikly to be accurate without related data)",
    )
    expected_non_financial_benefits: str = Field(
        ...,
        description="Expected non-financial benefits of the project (e.g., 'Improved customer satisfaction', 'Increased employee morale')",
    )


class TargetRiskReductionItemSchema(BaseModel):
    """Schema for a single 'For fixing this root cause' or 'Related asset' block."""

    risk_name: str
    root_cause: List[str] = Field(
        ...,
        description="List of root causes (e.g., 'Supplier A', 'Supplier B')",
    )
    related_asset: List[str] = Field(
        ...,
        description="List of related assets (e.g., 'Supplier A', 'Supplier B')",
    )

    target_risk_likelihood: Literal[
        "1: Rare", "2: Unlikely", "3: Moderate", "4: Likely", "5: Certain"
    ]  # Example values
    target_risk_impact: Literal[
        "1: Minor", "2: Moderate", "3: Significant", "4: Major", "5: Catastrophic"
    ]  # Example values


class Control(BaseModel):
    """Overall schema for the entire response dictionary."""

    plan_detail: PlanDetailSchema
    timeline_budget: TimelineBudgetSchema
    target_risk_reduction: List[
        TargetRiskReductionItemSchema
    ]  # List because there are two such blocks
    # this is optional in future feature we can access the support documents from the user
    support_documents: List[Optional[FileSchema]] = Field(
        ...,
        description=(
            "List of support documents. This field is optional and may be empty. "
            "Need to access support documents from the user."
        ),
    )
    task_management: List[TaskSchema]  # List of tasks


class MitigationPlan(Control): ...


class ExistingControl(Control): ...


class MitigationPlanResponse(MitigationPlan): ...


class MitigationPlanRequest(RiskRecommendationRequest):
    """Request schema for mitigation plan generation. include data in RiskRecommendationRequest but have optional data of existing control"""

    existing_controls: Optional[List[ExistingControl]] = Field(
        None, description="List of existing controls to consider"
    )
