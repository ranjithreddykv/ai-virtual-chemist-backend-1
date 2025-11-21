from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rdkit import Chem
import re
from typing import Optional, List, Tuple

MODEL_NAME = "sagawa/ReactionT5v2-forward"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"✓ Model loaded on {device}")


def canonicalize_smiles(smiles: str, use_canonical: bool = True) -> Optional[str]:
    """
    Canonicalize SMILES using RDKit.
    Returns None if invalid SMILES.
    """
    try:
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if use_canonical:
            return Chem.MolToSmiles(mol, canonical=True)
        return Chem.MolToSmiles(mol)
    except:
        return None


def prepare_input(reactants: str, reagents: str = "") -> str:
    """
    Prepare input in the correct format for ReactionT5v2-forward model.
    Format: REACTANT:{reactants}REAGENT:{reagents}
    
    CRITICAL: Keep reactants as-is, only strip whitespace.
    The model was trained on specific SMILES representations.
    """
    reactants_clean = reactants.strip()
    
    # Canonicalize each reactant component separately
    if '.' in reactants_clean:
        parts = reactants_clean.split('.')
        canonical_parts = []
        for part in parts:
            canon = canonicalize_smiles(part)
            if canon:
                canonical_parts.append(canon)
            else:
                canonical_parts.append(part.strip())
        reactants_clean = '.'.join(canonical_parts)
    else:
        canon = canonicalize_smiles(reactants_clean)
        if canon:
            reactants_clean = canon
    
    # Format according to ReactionT5v2 requirements
    if reagents and reagents.strip():
        reagents_clean = reagents.strip()
        input_text = f"REACTANT:{reactants_clean}REAGENT:{reagents_clean}"
    else:
        input_text = f"REACTANT:{reactants_clean}REAGENT:"
    
    return input_text


def clean_and_validate_smiles(smiles: str, keep_largest: bool = True) -> str:
    """
    Clean, validate, and canonicalize SMILES.
    Handles common artifacts from T5 models.
    """
    # Remove spaces
    smiles = smiles.strip().replace(" ", "")
    
    # Remove trailing dots
    smiles = smiles.rstrip(".")
    
    # If multiple molecules separated by dots
    if "." in smiles:
        fragments = smiles.split(".")
        
        # Filter out very small fragments and common ions
        valid_fragments = []
        for f in fragments:
            f = f.strip()
            # Skip empty, very small, or simple ion fragments
            if len(f) <= 2 or re.match(r"^\[(?:Na|K|Li|Cl|Br|I|H)\+?\-?\]$", f):
                continue
            # Validate with RDKit
            if canonicalize_smiles(f):
                valid_fragments.append(f)
        
        if not valid_fragments:
            fragments = [f for f in smiles.split(".") if len(f) > 2]
            if not fragments:
                return smiles
            valid_fragments = fragments
        
        if keep_largest:
            # Take the longest fragment as the main product
            smiles = max(valid_fragments, key=len)
        else:
            smiles = ".".join(valid_fragments)
    
    # Try to canonicalize
    canonical = canonicalize_smiles(smiles)
    if canonical:
        return canonical
    
    # If canonicalization fails, try removing common artifacts
    cleaned = re.sub(r"\[(?:Na|K|Li|Mg|Ca|Zn|Al|Fe|Cu|OH|H)\+?\-?\]\.?", "", smiles)
    canonical = canonicalize_smiles(cleaned)
    if canonical:
        return canonical
    
    # Return original if nothing worked
    return smiles


def predict_product(
    reactants: str,
    reagents: str = "",
    num_beams: int = 10,
    max_length: int = 512,
    temperature: float = 1.0,
    num_return: int = 1
) -> str:
    """
    Predict reaction product from reactants and reagents.
    
    Args:
        reactants: SMILES string of reactants (use . to separate multiple)
        reagents: SMILES or text description of reagents/conditions
        num_beams: Number of beams for beam search (higher = better but slower)
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature (1.0 = default)
        num_return: Number of predictions to consider (returns best)
    
    Returns:
        Cleaned and validated product SMILES
    """
    # Prepare input
    input_text = prepare_input(reactants, reagents)
    print(f"[DEBUG] Input to model: {input_text}")
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=min(num_return, num_beams),
            max_length=max_length,
            early_stopping=True,
            temperature=temperature,
            do_sample=False,
            return_dict_in_generate=True
        )
    
    # Decode and process all outputs
    predictions = []
    for seq in outputs.sequences:
        predicted_smiles = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"[DEBUG] Raw model output: {predicted_smiles}")
        cleaned_smiles = clean_and_validate_smiles(predicted_smiles)
        predictions.append(cleaned_smiles)
    
    # Return the first (best) prediction
    return predictions[0] if predictions else ""


def predict_product_with_alternatives(
    reactants: str,
    reagents: str = "",
    num_return: int = 5,
    num_beams: int = 10
) -> List[Tuple[str, str]]:
    """
    Predict multiple alternative products with their raw outputs.
    
    Args:
        reactants: SMILES string of reactants
        reagents: SMILES or text description of reagents/conditions
        num_return: Number of alternative predictions to return
        num_beams: Number of beams for beam search
    
    Returns:
        List of tuples (cleaned_smiles, raw_output)
    """
    input_text = prepare_input(reactants, reagents)
    print(f"[DEBUG] Input to model: {input_text}")
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=min(num_return, num_beams),
            max_length=512,
            early_stopping=True,
            return_dict_in_generate=True
        )
    
    # Decode all predictions
    predictions = []
    seen = set()
    
    for seq in outputs.sequences:
        raw_smiles = tokenizer.decode(seq, skip_special_tokens=True)
        cleaned = clean_and_validate_smiles(raw_smiles)
        
        # Only add unique valid predictions
        if cleaned and cleaned not in seen:
            if canonicalize_smiles(cleaned):
                predictions.append((cleaned, raw_smiles))
                seen.add(cleaned)
    
    return predictions


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Test 1: Nucleophilic Substitution (SN2)")
    print("="*60)
    reactants1 = "CCBr.C#N"  # Changed from CN to C#N
    product1 = predict_product(reactants1, "", num_beams=10)
    print(f"Reactants: {reactants1}")
    print(f"Expected: CCC#N (propanenitrile)")
    print(f"Predicted: {product1}")
    print()
    
    # Try with sodium cyanide
    print("="*60)
    print("Test 1b: With sodium cyanide")
    print("="*60)
    reactants1b = "CCBr.[Na+].[C-]#N"
    product1b = predict_product(reactants1b, "", num_beams=10)
    print(f"Reactants: {reactants1b}")
    print(f"Expected: CCC#N (propanenitrile)")
    print(f"Predicted: {product1b}")
    print()
    
    print("="*60)
    print("Test 2: Simple Ester Hydrolysis")
    print("="*60)
    reactants2 = "CC(=O)OC.O"
    reagents2 = "[OH-]"  # Adding base catalyst
    product2 = predict_product(reactants2, reagents2)
    print(f"Reactants: {reactants2}")
    print(f"Reagents: {reagents2}")
    print(f"Predicted: {product2}")
    print()
    
    print("="*60)
    print("Test 3: Top 5 Alternative Predictions")
    print("="*60)
    alternatives = predict_product_with_alternatives("CCBr.C#N", "", num_return=5)
    for i, (cleaned, raw) in enumerate(alternatives, 1):
        print(f"{i}. Cleaned: {cleaned}")
        print(f"   Raw: {raw}")
        print()