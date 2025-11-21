from dotenv import load_dotenv
from pathlib import Path
import os
from groq import Groq
from app.config import groq_client as client

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY missing in .env!")

groq_client = Groq(api_key=GROQ_API_KEY)

def explain_forward_reaction(reactants, reagents, product):
    prompt = f"""
You are an organic chemistry professor.
Explain the forward reaction clearly in 4–7 short sentences.

Reactants:
{reactants}

Reagents:
{reagents}

Predicted Product:
{product}

Explain:
• Which functional groups react
• What key bond is formed
• Why the reagents enable the transformation
• What the overall reaction accomplishes

Keep it simple, clear, and beginner-friendly.
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return res.choices[0].message.content

def explain_retro_reaction(product: str, reactants: list[str]):
    reactants_str = " + ".join(reactants)
    return (
        f"To synthesize {product}, a retrosynthetic analysis identifies the following "
        f"precursor(s): {reactants_str}. The reaction likely proceeds by breaking key "
        f"bonds in {product} and identifying synthetically accessible fragments. "
        f"Each predicted reactant represents a feasible disconnection step that aligns "
        f"with known organic synthesis strategies."
    )


def explain_yield_reason(reactants, reagents, product, predicted_yield):
    return (
        f"The predicted yield for converting {reactants} into {product} "
        f"under the conditions {reagents} is {predicted_yield}%. This estimate "
        f"reflects steric hindrance, possible side reactions, stability of "
        f"intermediates, and overall reaction compatibility."
    )

def suggest_yield_improvements(reactants, reagents, product, predicted_yield):
    improvements = [
        "Optimize reaction temperature (slightly higher or lower).",
        "Try alternative solvent systems to stabilize intermediates.",
        "Use a more selective catalyst or base.",
        "Increase reaction time if conversion is incomplete.",
        "Reduce steric hindrance by modifying substituents.",
        "Remove moisture and oxygen from reaction environment."
    ]

    return (
        f"To improve the yield of {product}, consider the following adjustments:\n"
        + "\n".join(f"- {i}" for i in improvements)
    )

from app.ml.sagawa_yield import predict_yield


def search_better_yield_conditions(
    reactants: str,
    reagents: str,
    product: str,
    base_yield: float,
    threshold: float = 2.0,   # only show if ≥ 2% better
) -> list[dict]:
    """
    Generate alternative conditions, score them with the yield model,
    and return only those that beat the current predicted yield.
    """

    # 1) Build simple candidate condition variants
    base_reagents = reagents or ""

    candidates = set()
    candidates.add(base_reagents)  # original

    # some "typical" base / solvent tweaks
    common_conditions = [
        "NaOH.H2O",
        "NaOH.H2O.EtOH",
        "K2CO3.H2O",
        "KOH.H2O",
        "Na2CO3.H2O",
        "NaOH.MeOH",
        "NaOH.DMSO",
    ]

    for cond in common_conditions:
        candidates.add(cond)

    improved = []

    for cond in candidates:
        try:
            y = predict_yield(reactants, cond, product)
        except Exception:
            continue

        if y is None:
            continue

        if y >= base_yield + threshold:
            improved.append(
                {
                    "reactants": reactants,
                    "reagents": cond,
                    "product": product,
                    "predicted_yield": float(y),
                    "delta_yield": float(y - base_yield),
                    "rationale": (
                        "Condition variant suggested from a predefined set. "
                        "Yield estimated by the same ML model."
                    ),
                }
            )

    # sort best → worst
    improved.sort(key=lambda x: x["predicted_yield"], reverse=True)
    return improved[:5]  # top 5 suggestions
