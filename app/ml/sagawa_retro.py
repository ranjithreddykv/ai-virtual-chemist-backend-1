from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rdkit import Chem
import re
from typing import Optional, List, Tuple

MODEL_NAME = "sagawa/ReactionT5v2-retrosynthesis-USPTO_50k"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"✓ Retrosynthesis model loaded on {device}")


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


def clean_and_validate_reactants(smiles: str) -> str:
    """
    Clean and validate predicted reactants SMILES.
    Handles multiple reactants separated by dots.
    """
    # Remove spaces
    smiles = smiles.strip().replace(" ", "")
    
    # Remove trailing dots
    smiles = smiles.rstrip(".")
    
    # If multiple reactants separated by dots
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
            canon = canonicalize_smiles(f)
            if canon:
                valid_fragments.append(canon)
        
        if not valid_fragments:
            # If all were filtered out, try to keep larger fragments
            fragments = [f for f in smiles.split(".") if len(f) > 2]
            if fragments:
                valid_fragments = fragments
            else:
                return smiles
        
        # Return all valid reactants joined by dots
        return ".".join(valid_fragments)
    
    # Single molecule - try to canonicalize
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


def prepare_product_input(product: str) -> str:
    """
    Prepare product SMILES for retrosynthesis model.
    Canonicalize to ensure consistent input format.
    """
    product_clean = product.strip()
    
    # Try to canonicalize
    canonical = canonicalize_smiles(product_clean)
    if canonical:
        return canonical
    
    return product_clean


def predict_reactants(
    product: str,
    num_beams: int = 5,
    max_length: int = 512,
    num_return: int = 1
) -> str:
    """
    Predict reactants from product SMILES using retrosynthesis model.
    
    Args:
        product: SMILES string of the product molecule
        num_beams: Number of beams for beam search (higher = better but slower)
        max_length: Maximum length of generated sequence
        num_return: Number of predictions to consider (returns best)
    
    Returns:
        Cleaned and validated reactants SMILES (separated by dots)
    """
    # Prepare and canonicalize input
    product_input = prepare_product_input(product)
    print(f"[DEBUG] Input product: {product_input}")
    
    # Tokenize
    inputs = tokenizer(
        product_input,
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
            return_dict_in_generate=True
        )
    
    # Decode and process all outputs
    predictions = []
    for seq in outputs.sequences:
        predicted_smiles = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"[DEBUG] Raw model output: {predicted_smiles}")
        cleaned_smiles = clean_and_validate_reactants(predicted_smiles)
        predictions.append(cleaned_smiles)
    
    # Return the first (best) prediction
    return predictions[0] if predictions else ""


def predict_reactants_with_alternatives(
    product: str,
    num_return: int = 5,
    num_beams: int = 10
) -> List[Tuple[str, str]]:
    """
    Predict multiple alternative retrosynthetic routes.
    
    Args:
        product: SMILES string of the product molecule
        num_return: Number of alternative predictions to return
        num_beams: Number of beams for beam search
    
    Returns:
        List of tuples (cleaned_reactants, raw_output)
    """
    product_input = prepare_product_input(product)
    print(f"[DEBUG] Input product: {product_input}")
    
    inputs = tokenizer(
        product_input,
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
        cleaned = clean_and_validate_reactants(raw_smiles)
        
        # Only add unique valid predictions
        if cleaned and cleaned not in seen:
            # Verify it's valid SMILES
            valid = True
            if '.' in cleaned:
                # Check each component
                valid = all(canonicalize_smiles(part) for part in cleaned.split('.'))
            else:
                valid = canonicalize_smiles(cleaned) is not None
            
            if valid:
                predictions.append((cleaned, raw_smiles))
                seen.add(cleaned)
    
    return predictions


def validate_retrosynthesis(product: str, predicted_reactants: str) -> bool:
    """
    Validate that predicted reactants are chemically valid.
    
    Args:
        product: Original product SMILES
        predicted_reactants: Predicted reactants SMILES
    
    Returns:
        True if reactants are valid molecules
    """
    try:
        # Check product is valid
        product_mol = Chem.MolFromSmiles(product)
        if product_mol is None:
            return False
        
        # Check each reactant is valid
        if '.' in predicted_reactants:
            reactants = predicted_reactants.split('.')
            for reactant in reactants:
                mol = Chem.MolFromSmiles(reactant)
                if mol is None:
                    return False
        else:
            mol = Chem.MolFromSmiles(predicted_reactants)
            if mol is None:
                return False
        
        return True
    except:
        return False


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Test 1: Simple Retrosynthesis - Propanenitrile")
    print("="*60)
    product1 = "CCC#N"  # Propanenitrile
    reactants1 = predict_reactants(product1, num_beams=10)
    print(f"Product: {product1}")
    print(f"Expected reactants: CCBr.C#N (or similar)")
    print(f"Predicted reactants: {reactants1}")
    print(f"Valid: {validate_retrosynthesis(product1, reactants1)}")
    print()
    
    print("="*60)
    print("Test 2: Ester Retrosynthesis")
    print("="*60)
    product2 = "CC(=O)OCC"  # Ethyl acetate
    reactants2 = predict_reactants(product2, num_beams=10)
    print(f"Product: {product2}")
    print(f"Expected reactants: CC(=O)O.CCO (or similar)")
    print(f"Predicted reactants: {reactants2}")
    print(f"Valid: {validate_retrosynthesis(product2, reactants2)}")
    print()
    
    print("="*60)
    print("Test 3: Amide Retrosynthesis")
    print("="*60)
    product3 = "CC(=O)NC"  # N-methylacetamide
    reactants3 = predict_reactants(product3, num_beams=10)
    print(f"Product: {product3}")
    print(f"Expected reactants: CC(=O)O.CN (or CC(=O)Cl.CN)")
    print(f"Predicted reactants: {reactants3}")
    print(f"Valid: {validate_retrosynthesis(product3, reactants3)}")
    print()
    
    print("="*60)
    print("Test 4: Top 5 Alternative Routes for Propanenitrile")
    print("="*60)
    alternatives = predict_reactants_with_alternatives("CCC#N", num_return=5)
    for i, (cleaned, raw) in enumerate(alternatives, 1):
        print(f"{i}. Cleaned: {cleaned}")
        print(f"   Raw: {raw}")
        print(f"   Valid: {validate_retrosynthesis('CCC#N', cleaned)}")
        print()