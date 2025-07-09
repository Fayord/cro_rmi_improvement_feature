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
)


# Initialize FastAPI app
app = FastAPI(
    title="Risk Recommendation API",
    description="API for recommending risks to assess based on existing risks and user context",
    version="1.0.0",
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
