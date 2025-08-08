"""
FastAPI application for risk recommendation service.
"""

from datetime import datetime

from json import load
from typing import List, Optional, Literal
import uuid
from typing import Dict
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from schemas import (
    RiskRecommendationRequest,
    RiskDataWithTags,
    ErrorResponse,
    MitigationPlanResponse,
    MitigationPlanRequest,
    PlanDetailSchema,
    TimelineBudgetSchema,
    TargetRiskReductionItemSchema,
    FileSchema,
    TaskSchema,
    RiskRecommendationAssessmentResponse,
    RiskRecommendationMitigationPlanResponse,
    GraphDataRetrievalRequest,
    GraphDataRetrievalResponse,
    ClusterMitigationPlanRequest,
    LatestGraphIdRequest,
    LatestGraphIdResponse,
)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
from data_processor.create_graph_data_library import (
    GraphDataLibrary,
    CompanyGraphData,
    RiskDataWithEmbedding,
    RiskData,
    EdgeData,
    RiskOverlayData,
)
from test_jwt import verify_access_token_fastapi
from core.models import Process, RootCause
from time import time

from risk_to_assess import (
    recommend_risk_to_assesses,
    convert_existing_risk_to_risk_data,
    create_company_graph_data,
    save_company_graph_data,
    load_graph_data_library,
)
from risk_to_mitigate import (
    get_source_and_central_risk,
    update_tags_risk_data,
    remove_existing_risk_with_mitigation_plan,
)
from traceback import print_exc

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


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Risk Recommendation API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "risk-recommendation-api",
    }


def retrieve_graph_data(graph_id: str) -> GraphDataRetrievalResponse:
    # TODO: Implement the logic to retrieve graph data
    return GraphDataRetrievalResponse(
        graph_visualize_data={},
        cluster_of_risks=[],
    )


@app.post(
    "/retrieve_graph_data",
    response_model=GraphDataRetrievalResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def retrieve_graph_data_api(
    request: GraphDataRetrievalRequest,
    payload: Dict = Depends(verify_access_token_fastapi),
):
    """Retrieve graph data for a given graph identifier."""
    return retrieve_graph_data(request.graph_id)


@app.post(
    "/recommend_risk_to_assess",
    response_model=RiskRecommendationAssessmentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def recommend_risks_to_assess_api(
    request: RiskRecommendationRequest,
    payload: Dict = Depends(verify_access_token_fastapi),
):
    """
    Expected to use all data in the same year and quarter
    Generate risk recommendations for assessment based on existing risks and user context.

    This endpoint analyzes the provided existing risks and user information to generate
    personalized risk recommendations that should be assessed.
    """
    if request.data_level not in ["user", "company"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data level must be user or company for now",
        )
    try:
        # Validate input
        if not request.existing_risks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one existing risk must be provided",
            )
        # if request.data_level is in ["user", "company"], check all company is the same
        if request.data_level in ["user", "company"]:
            if not all(
                risk.company_id == request.existing_risks[0].company_id
                for risk in request.existing_risks
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All company must be the same",
                )
        existing_risk_list = request.existing_risks
        # for user level, we don't need to save and create graph
        # for company level, we need to save data

        graph_data_library = load_graph_data_library()
        if request.data_level == "user":
            # convert existing_risk to risk_data
            risk_data_list = convert_existing_risk_to_risk_data(existing_risk_list)
        elif request.data_level == "company":
            # for company level we can do it in 2 phrase by
            # phrase one
            # 1. get embedding of input risk data
            # phrase two
            # 1. after return result we need to update the graph data with new risk data
            # 2. save the graph data to graph data library

            company_id = request.existing_risks[0].company_id
            company_graph_id = f"{company_id}|embedding_risk_desc_catalog|oneway_run"
            risk_data_list = convert_existing_risk_to_risk_data(existing_risk_list)

            start_time = time()
            company_graph_data = create_company_graph_data(
                risk_data_list,
                company_name=company_id,
                embedding_key="embedding_risk_desc_catalog",
                classify_model_name="gpt-4.1-mini",
                high_priority_search_space=4.0,
                high_priority_atmost_number_edges=3,
                relation_process="oneway_run",
                graph_data_library=graph_data_library,
            )
            end_time = time()
            print(
                f"Time taken to create company graph data: {end_time - start_time} seconds"
            )
            start_time = time()
            save_company_graph_data(
                company_graph_data, company_graph_id, graph_data_library
            )
            end_time = time()
            print(
                f"Time taken to save company graph data: {end_time - start_time} seconds"
            )

        # pass list risk_data to recommend_risk_to_assesses
        all_recommended_risks = recommend_risk_to_assesses(
            risk_data_list, request.year_quarter, graph_data_library
        )
        return RiskRecommendationAssessmentResponse(
            id=request.id, recommendations=all_recommended_risks
        )
    except (ValueError, AttributeError, UnboundLocalError) as e:
        print_exc()
        # return mockup data for now
        return RiskRecommendationAssessmentResponse(
            id=request.id,
            recommendations=[
                RiskData(
                    id="risk_001",
                    label="Risk 1",
                    risk="Risk 1",
                    risk_id="risk_001",
                    risk_cat="Operational",
                    risk_level=3,
                    risk_score=3,
                    risk_likelihood=3,
                    risk_impact=3,
                    company_id="company_001",
                    process=[
                        Process(
                            id="process_001",
                            name="Process 1",
                            description="Process 1 description",
                        )
                    ],
                    risk_desc="Risk 1 description",
                    rootcause=[
                        RootCause(
                            id="rootcause_001",
                            name="Rootcause 1",
                            description="Rootcause 1 description",
                        )
                    ],
                    process_summary="Process 1 summary",
                    rootcause_summary="Rootcause 1 summary",
                    risk_desc_summary="Risk 1 description summary",
                ),
                RiskData(
                    id="risk_002",
                    label="Risk 2",
                    risk="Risk 2",
                    risk_id="risk_002",
                    risk_cat="Operational",
                    risk_level=3,
                    risk_score=3,
                    risk_likelihood=3,
                    risk_impact=3,
                    company_id="company_001",
                    process=[
                        Process(
                            id="process_002",
                            name="Process 2",
                            description="Process 2 description",
                        )
                    ],
                    risk_desc="Risk 2 description",
                    rootcause=[
                        RootCause(
                            id="rootcause_002",
                            name="Rootcause 2",
                            description="Rootcause 2 description",
                        )
                    ],
                    process_summary="Process 2 summary",
                    rootcause_summary="Rootcause 2 summary",
                    risk_desc_summary="Risk 2 description summary",
                ),
            ],
        )
    except HTTPException:
        print_exc()
        raise
    except Exception as e:
        print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating assessment recommendations: {str(e)}",
        )


@app.post(
    "/recommend_risk_to_mitigate",
    response_model=RiskRecommendationMitigationPlanResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def recommend_risks_to_mitigate_api(
    request: RiskRecommendationRequest,
    payload: Dict = Depends(verify_access_token_fastapi),
):
    """
    Generate risk recommendations for mitigation based on existing risks and user context.

    This endpoint analyzes the provided existing risks and user information to generate
    personalized risk recommendations that should be mitigated.
    """
    if request.data_level not in ["company"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data level must be company for now",
        )
    try:
        # Validate input
        if not request.existing_risks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one existing risk must be provided",
            )
        graph_data_library = load_graph_data_library()
        existing_risk_list = request.existing_risks
        if request.data_level == "company":
            if not all(
                risk.company_id == request.existing_risks[0].company_id
                for risk in request.existing_risks
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All company must be the same",
                )
            # create new graph data
            # then find the source and central risk from graph data
            # save the updated graph data
            company_id = request.existing_risks[0].company_id
            company_graph_id = f"{company_id}|embedding_risk_desc_catalog|oneway_run"
            risk_data_list = convert_existing_risk_to_risk_data(existing_risk_list)
            company_graph_data = create_company_graph_data(
                risk_data_list,
                company_name=company_id,
                embedding_key="embedding_risk_desc_catalog",
                classify_model_name="gpt-4.1-mini",
                high_priority_search_space=4.0,
                high_priority_atmost_number_edges=3,
                relation_process="oneway_run",
                graph_data_library=graph_data_library,
            )

            save_company_graph_data(
                company_graph_data, company_graph_id, graph_data_library
            )

            source_risks, central_risks = get_source_and_central_risk(
                company_graph_data
            )
            # then update tags (it can be multiple tags)
            # then remove the risk that already have mitigation plan
            risk_data_dict = {
                "is_source_risk": source_risks,
                "is_central_risk": central_risks,
            }
            recommendations_risks: List[RiskDataWithTags] = update_tags_risk_data(
                risk_data_dict
            )
            recommendations_risks: List[RiskDataWithTags] = (
                remove_existing_risk_with_mitigation_plan(
                    recommendations_risks, existing_risk_list
                )
            )

        response = RiskRecommendationMitigationPlanResponse(
            id=request.id,
            recommendations_with_tags=recommendations_risks,
            graph_id=uuid.uuid4().hex[:8],
        )

        return response

    except HTTPException:
        print_exc()
        raise
    except Exception as e:
        print_exc()
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
)
async def generate_mitigation_plan(
    request: MitigationPlanRequest, payload: Dict = Depends(verify_access_token_fastapi)
):
    """
    Future Feature: Generate a mitigation plan for a given risks.
    """
    return _generate_mock_mitigation_plan(request)


@app.post(
    "/generate_cluster_mitigation_plan",
    response_model=List[MitigationPlanResponse],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def generate_cluster_mitigation_plan(
    request: List[ClusterMitigationPlanRequest],
    payload: Dict = Depends(verify_access_token_fastapi),
):
    """
    Generate a mitigation plan for a given risk. anything with a dropdown/Literal type eg. priority_level, status, etc. Need to review the schema later
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
    request: ClusterMitigationPlanRequest,
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


def _generate_mock_mitigation_recommendations(
    request: RiskRecommendationRequest,
) -> List[RiskDataWithTags]:
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
        RiskDataWithTags(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Supply Chain Disruption",
        ),
        RiskDataWithTags(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Financial Market Volatility",
        ),
        RiskDataWithTags(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Operational Inefficiency",
        ),
        RiskDataWithTags(
            risk_id=f"risk_{uuid.uuid4().hex[:8]}",
            risk_name="Technology Infrastructure Risk",
        ),
        RiskDataWithTags(
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


def get_latest_graph_id_from_library(
    data_level: Literal["company", "country", "multi_country"],
    id: str,
) -> Optional[str]:
    """Retrieves the latest graph ID from the GraphDataLibrary based on data level and ID."""
    graph_data_library: GraphDataLibrary = load_graph_data_library()

    # The graph IDs are stored in the format "company_id|embedding_key|relation_process"
    # We need to filter based on data_level and then find the latest one (e.g., by timestamp or a similar mechanism)
    # For simplicity, let's assume the IDs are already in a sortable format or we can extract a timestamp.
    # For now, let's assume a direct match for the 'id' part of the graph_id and that the latest is the one with highest version, e.g. timestamp

    # collect all graph ids that match the data_level and id
    matching_graph_ids = []
    for graph_id_key in graph_data_library.company_graph_datas.keys():
        parts = graph_id_key.split("|")
        if len(parts) >= 1 and parts[0] == id:  # Assuming id is the first part for now
            matching_graph_ids.append(graph_id_key)

    if not matching_graph_ids:
        return None

    # Sort to get the latest. This assumes a convention like timestamp in the graph_id.
    # If no timestamp, then a more complex logic is needed, or a simple alphabetical sort.
    # For this implementation, we'll assume a temporal component or version is implicitly sortable.
    # For example, if graph_id is 'company_id|embedding_key|timestamp'
    # The exact sorting logic might need to be refined based on actual graph ID structure.
    return sorted(matching_graph_ids)[
        -1
    ]  # Return the latest one alphabetically/lexicographically


@app.post(
    "/retrieve_latest_graph_id",
    response_model=LatestGraphIdResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def retrieve_latest_graph_id_api(
    request: LatestGraphIdRequest, payload: Dict = Depends(verify_access_token_fastapi)
):
    """Retrieve the latest graph ID for a given company/country/multi_country."""
    try:
        # TODO: Implement the logic to retrieve the latest graph ID
        latest_graph_id = "DUMMY_GRAPH_ID"

        if latest_graph_id:
            return LatestGraphIdResponse(
                id=request.id,
                graph_id=latest_graph_id,
                message=f"Successfully retrieved latest graph ID for {request.data_level} {request.id}.",
            )
        else:
            return LatestGraphIdResponse(
                id=request.id,
                graph_id=None,
                message=f"No graph ID found for {request.data_level} {request.id}.",
            )
    except Exception as e:
        print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving latest graph ID: {str(e)}",
        )


@app.get("/protected")
def protected_route(payload: Dict = Depends(verify_access_token_fastapi)):
    """
    A protected endpoint that requires a valid access token.
    The `payload` variable will contain the decoded JWT data.
    """
    return {"message": "Access granted", "user_info": payload}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7900, reload=False, log_level="info")
