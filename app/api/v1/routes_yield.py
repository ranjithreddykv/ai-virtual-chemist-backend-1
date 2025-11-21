from fastapi import APIRouter, HTTPException
from app.schemas.yield_prediction_schema import YieldRequest
from app.ml.sagawa_yield import predict_yield
from app.utils.mechanism_visualizer_forward import generate_3d_molblock
from app.utils.mechanism_explainer import (
    explain_yield_reason,
    suggest_yield_improvements,
    search_better_yield_conditions,
)

router = APIRouter(prefix="/yield", tags=["Reaction Yield Prediction"])


@router.post("/predict")
async def predict_yield_basic(request: YieldRequest):
    """
    Old/basic endpoint: returns only predicted yield.
    """
    try:
        result = predict_yield(request.reactants, request.reagents, request.product)
        return {"predicted_yield": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_full")
async def predict_yield_full(request: YieldRequest):
    try:
        base_yield = predict_yield(
            request.reactants, request.reagents, request.product
        )

        reactants_list = request.reactants.split(".")
        reactants_3d = [generate_3d_molblock(r) for r in reactants_list]
        product_3d = generate_3d_molblock(request.product)

        explanation = explain_yield_reason(
            request.reactants, request.reagents, request.product, base_yield
        )

        text_improvement = suggest_yield_improvements(
            request.reactants, request.reagents, request.product, base_yield
        )

        # 🔥 NEW: actually search for better conditions using the model
        improved_candidates = search_better_yield_conditions(
            request.reactants,
            request.reagents,
            request.product,
            base_yield,
        )

        return {
            "reactants": request.reactants,
            "reagents": request.reagents,
            "product": request.product,
            "reactants_list": reactants_list,

            "predicted_yield": base_yield,
            "reactants_3d": reactants_3d,
            "product_3d": product_3d,

            "teacher_explanation": explanation,
            "yield_improvement": text_improvement,
            "yield_improvement_structured": improved_candidates,  # 👈 NEW
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    """
    Full yield predictor with:
    - Predicted yield
    - 3D reactants and product
    - LLM reasoning (why yield is good/bad)
    - LLM improvement suggestions
    """

    try:
        # Yield ML prediction
        y = predict_yield(request.reactants, request.reagents, request.product)

        # 3D visualizations
        reactants_list = request.reactants.split(".")
        react_3d = [generate_3d_molblock(r) for r in reactants_list]
        product_3d = generate_3d_molblock(request.product)

        # LLM reasoning
        explanation = explain_yield_reason(
            request.reactants,
            request.reagents,
            request.product,
            y
        )

        # Improvement suggestions
        improvements = suggest_yield_improvements(
            request.reactants,
            request.reagents,
            request.product,
            y
        )

        return {
            "reactants": request.reactants,
            "reagents": request.reagents,
            "product": request.product,

            "predicted_yield": y,

            "reactants_list": reactants_list,
            "reactants_3d": react_3d,
            "product_3d": product_3d,

            "teacher_explanation": explanation,
            "yield_improvement": improvements
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
