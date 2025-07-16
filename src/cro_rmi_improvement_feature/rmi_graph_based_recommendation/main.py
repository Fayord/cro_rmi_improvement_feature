"""
FastAPI application for risk recommendation service.
"""

from datetime import datetime
from typing import List
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from schemas import (
    RiskRecommendationRequest,
    RiskRecommendationResponse,
    RecommendedRisk,
    ErrorResponse,
    MitigationPlanRequest,
    MitigationPlanResponse,
    PlanDetailSchema,
    TimelineBudgetSchema,
    TargetRiskReductionItemSchema,
    FileSchema,
    TaskSchema,
)


# Initialize FastAPI app
app = FastAPI(
    title="Risk Recommendation API",
    description="API for recommending risks to assess based on existing risks and user context",
    version="1.0.0",
    root_path="/rmi_graph_based_recommendation",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"message": "Risk Recommendation API is running", "status": "healthy"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "risk-recommendation-api",
    }


@app.post(
    "/recommend_risk_to_assess",
    response_model=RiskRecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    tags=["Recommendations"],
)
async def recommend_risks_to_assess(request: RiskRecommendationRequest):
    """
    Generate risk recommendations for assessment based on existing risks and user context.

    This endpoint analyzes the provided existing risks and user information to generate
    personalized risk recommendations that should be assessed.
    """
    try:
        # Validate input
        if not request.existing_risks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one existing risk must be provided",
            )

        # TODO: Integrate with your existing risk assessment logic
        # For now, returning mock data
        recommendations = _generate_mock_assessment_recommendations(request)

        response = RiskRecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating assessment recommendations: {str(e)}",
        )


@app.post(
    "/recommend_risk_to_mitigate",
    response_model=RiskRecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    tags=["Recommendations"],
)
async def recommend_risks_to_mitigate(request: RiskRecommendationRequest):
    """
    Generate risk recommendations for mitigation based on existing risks and user context.

    This endpoint analyzes the provided existing risks and user information to generate
    personalized risk recommendations that should be mitigated.
    """
    try:
        # Validate input
        if not request.existing_risks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one existing risk must be provided",
            )

        # TODO: Integrate with your existing risk mitigation logic
        # For now, returning mock data
        recommendations = _generate_mock_mitigation_recommendations(request)

        response = RiskRecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating mitigation recommendations: {str(e)}",
        )


@app.post(
    "/generate_mitigation_plan",
    response_model=MitigationPlanResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    tags=["Generate"],
)
async def generate_mitigation_plan(request: MitigationPlanRequest):
    """
    Generate a mitigation plan for a given risk.
    """
    try:
        # Validate input
        if not request.risk_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Risk ID must be provided",
            )

        # TODO: Integrate with your existing risk mitigation logic
        # For now, returning mock data
        mitigation_plan = _generate_mock_mitigation_plan(request)

        return mitigation_plan

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating mitigation plan: {str(e)}",
        )


def _generate_mock_mitigation_plan(
    request: MitigationPlanRequest,
) -> MitigationPlanResponse:
    """
    Generate mock mitigation plan for a given risk.
    """
    # Extract risk information from request
    risk_name = (
        request.existing_risks[0].risk_name
        if request.existing_risks
        else "Supply Chain Disruption"
    )
    risk_id = (
        request.existing_risks[0].risk_id if request.existing_risks else "risk_001"
    )

    # Create Plan Detail
    plan_detail = PlanDetailSchema(
        mitigation_plan_name=f"Mitigation Plan for {risk_name}",
        mitigation_plan_objective=f"Reduce the likelihood and impact of {risk_name} through comprehensive risk management strategies",
        linked_risks=[risk_name, "Operational Risk", "Financial Risk"],
        priority_level="High",
        plan_owner="John Smith",
        plan_owner_email="john.smith@company.com",
    )

    # Create Timeline & Budget
    timeline_budget = TimelineBudgetSchema(
        start_date="2024-01-15",
        end_date="2024-06-30",
        expected_cost=150000.0,
        expected_financial_benefit=500000.0,
        expected_non_financial_benefits="Improved operational efficiency, enhanced stakeholder confidence, reduced regulatory exposure",
    )

    # Create Target Risk Reduction items
    target_risk_reduction = [
        TargetRiskReductionItemSchema(
            risk_name=risk_name,
            root_cause=[
                "Single source dependency",
                "Lack of supplier diversification",
                "Inadequate risk monitoring",
            ],
            related_asset=[
                "Primary Supplier A",
                "Secondary Supplier B",
                "Risk Management System",
            ],
            target_risk_likelihood="2: Unlikely",
            target_risk_impact="2: Moderate",
        ),
        TargetRiskReductionItemSchema(
            risk_name="Operational Risk",
            root_cause=[
                "Process inefficiencies",
                "Manual intervention required",
                "Lack of automation",
            ],
            related_asset=[
                "Process Management System",
                "Automation Tools",
                "Employee Training Program",
            ],
            target_risk_likelihood="3: Moderate",
            target_risk_impact="3: Significant",
        ),
    ]

    # Create Support Documents
    support_documents = [
        FileSchema(name="Risk Assessment Report.pdf", size="2.5 MB", status="uploaded"),
        FileSchema(
            name="Mitigation Strategy Document.docx", size="1.8 MB", status="uploaded"
        ),
        FileSchema(name="Budget Analysis.xlsx", size="856 KB", status="uploaded"),
    ]

    # Create Task Management
    task_management = [
        TaskSchema(
            pre_defined_task="Conduct comprehensive risk assessment",
            weighted_activity_percent="15",
            status="In Progress",
            task_owner_placeholder="Sarah Johnson",
            task_owner_department="Risk Management",
            task_owner_email="sarah.johnson@company.com",
            start_date="2024-01-15",
            end_date="2024-02-15",
        ),
        TaskSchema(
            pre_defined_task="Develop supplier diversification strategy",
            weighted_activity_percent="25",
            status="Not Started",
            task_owner_placeholder="Mike Chen",
            task_owner_department="Procurement",
            task_owner_email="mike.chen@company.com",
            start_date="2024-02-01",
            end_date="2024-03-31",
        ),
        TaskSchema(
            pre_defined_task="Implement risk monitoring system",
            weighted_activity_percent="30",
            status="Not Started",
            task_owner_placeholder="Lisa Wang",
            task_owner_department="IT",
            task_owner_email="lisa.wang@company.com",
            start_date="2024-03-01",
            end_date="2024-05-31",
        ),
        TaskSchema(
            pre_defined_task="Train staff on new procedures",
            weighted_activity_percent="20",
            status="Not Started",
            task_owner_placeholder="David Brown",
            task_owner_department="HR",
            task_owner_email="david.brown@company.com",
            start_date="2024-05-01",
            end_date="2024-06-15",
        ),
        TaskSchema(
            pre_defined_task="Conduct final review and validation",
            weighted_activity_percent="10",
            status="Not Started",
            task_owner_placeholder="Emily Davis",
            task_owner_department="Compliance",
            task_owner_email="emily.davis@company.com",
            start_date="2024-06-01",
            end_date="2024-06-30",
        ),
    ]

    return MitigationPlanResponse(
        plan_detail=plan_detail,
        timeline_budget=timeline_budget,
        target_risk_reduction=target_risk_reduction,
        support_documents=support_documents,
        task_management=task_management,
    )


def _generate_mock_assessment_recommendations(
    request: RiskRecommendationRequest,
) -> List[RecommendedRisk]:
    """
    Generate mock risk recommendations for assessment based on existing risks.

    In a real implementation, this would integrate with your existing
    risk assessment logic from the network_analyzer or other modules.
    """
    # Analyze existing risks to generate relevant recommendations
    existing_categories = set(risk.risk_category for risk in request.existing_risks)
    existing_risk_names = set(risk.risk_name for risk in request.existing_risks)

    # Mock risk data for assessment - replace with actual risk assessment logic
    mock_risks = [
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Cybersecurity Breach",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Regulatory Compliance Failure",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Data Privacy Violation",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Third-Party Vendor Risk",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Business Continuity Disruption",
        ),
    ]

    # Filter out risks that already exist
    filtered_risks = [
        risk for risk in mock_risks if risk.risk_name not in existing_risk_names
    ]

    # Return all filtered risks
    return filtered_risks


def _generate_mock_mitigation_recommendations(
    request: RiskRecommendationRequest,
) -> List[RecommendedRisk]:
    """
    Generate mock risk recommendations for mitigation based on existing risks.

    In a real implementation, this would integrate with your existing
    risk mitigation logic from the network_analyzer or other modules.
    """
    # Analyze existing risks to generate relevant recommendations
    existing_categories = set(risk.risk_category for risk in request.existing_risks)
    existing_risk_names = set(risk.risk_name for risk in request.existing_risks)

    # Mock risk data for mitigation - replace with actual risk mitigation logic
    mock_risks = [
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Supply Chain Disruption",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Financial Market Volatility",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Operational Inefficiency",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Technology Infrastructure Risk",
        ),
        RecommendedRisk(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Human Resource Risk",
        ),
    ]

    # Filter out risks that already exist
    filtered_risks = [
        risk for risk in mock_risks if risk.risk_name not in existing_risk_names
    ]

    # Return all filtered risks
    return filtered_risks


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7900, reload=True, log_level="info")
