def generate_retro_explanation(target, routes):
    """
    routes: a list of model-generated disconnections with reasoning
    """
    route_text = "\n".join([f"- {r}" for r in routes])

    prompt = f"""
You are an organic chemistry professor. Your task is to explain a RETROSYNTHETIC
ANALYSIS in a rigorous, structured, academically correct format.

Follow the same tone and discipline as the mechanism explanation system.

NO emojis.  
NO ASCII decorations.  
NO invented chemistry.  

----------------------------------------------------------
## Overview
Summarize the target molecule `{target}`, highlighting:
- Key functional groups
- Overall topology
- Synthetic challenges or features

----------------------------------------------------------
## Strategic Disconnections
Explain the model-identified strategic disconnections:

{route_text}

For each disconnection:
- Describe the bond being conceptually "broken".
- Identify which known reaction the disconnection corresponds to.
- Mention polarity reversal (umpolung) if relevant.
- Explain why the disconnection is considered strategic.

----------------------------------------------------------
## Synthons and Reagents
For each proposed disconnection:
- Identify the nucleophilic and electrophilic synthons.
- Suggest real reagents that correspond to these synthons.
- Provide 1–2 lines of justification.

----------------------------------------------------------
## Alternative Plausible Routes
Give 1–2 reasonable alternative synthetic ideas WITHOUT hallucination.

----------------------------------------------------------
## Final Insight
Provide a concise, elegant closing remark on why the identified route(s)
represent logical retrosynthetic strategies.

----------------------------------------------------------
"""

    return prompt
