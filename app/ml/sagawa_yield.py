import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, PreTrainedModel
from rdkit import Chem
import logging
from typing import Optional, Dict

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "sagawa/ReactionT5v2-yield"


class ReactionT5Yield(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)
        self.model.resize_token_embeddings(self.config.vocab_size)

        # Fully connected layers for yield regression
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.fc3 = nn.Linear(self.config.hidden_size // 2 * 2, self.config.hidden_size)
        self.fc4 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc5 = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        self._init_weights(self.fc4)
        self._init_weights(self.fc5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        encoder_outputs = self.model.encoder(**inputs)
        encoder_hidden_states = encoder_outputs[0]
        outputs = self.model.decoder(
            input_ids=torch.full(
                (inputs["input_ids"].size(0), 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=inputs["input_ids"].device,
            ),
            encoder_hidden_states=encoder_hidden_states,
        )
        last_hidden_states = outputs[0]
        output1 = self.fc1(last_hidden_states.view(-1, self.config.hidden_size))
        output2 = self.fc2(encoder_hidden_states[:, 0, :].view(-1, self.config.hidden_size))
        output = self.fc3(torch.hstack((output1, output2)))
        output = self.fc4(output)
        output = self.fc5(output)
        return output * 100  # Scale to percentage


# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ReactionT5Yield.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"✓ Yield prediction model loaded on {device}")


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


def prepare_reaction_input(reactants: str, reagents: str, product: str) -> str:
    """
    Prepare input in the correct format for ReactionT5v2-yield model.
    Format: REACTANT:{reactants}REAGENT:{reagents}PRODUCT:{product}
    
    Canonicalizes each component for consistency.
    """
    # Canonicalize reactants (may contain multiple molecules)
    reactants_clean = reactants.strip()
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
    
    # Handle reagents (may be empty, text, or SMILES)
    reagents_clean = reagents.strip() if reagents else ""
    if reagents_clean and not any(c.isalpha() and c.islower() for c in reagents_clean):
        # Looks like SMILES, try to canonicalize
        if '.' in reagents_clean:
            parts = reagents_clean.split('.')
            canonical_parts = []
            for part in parts:
                canon = canonicalize_smiles(part)
                if canon:
                    canonical_parts.append(canon)
                else:
                    canonical_parts.append(part.strip())
            reagents_clean = '.'.join(canonical_parts)
        else:
            canon = canonicalize_smiles(reagents_clean)
            if canon:
                reagents_clean = canon
    
    # Canonicalize product
    product_clean = product.strip()
    canon = canonicalize_smiles(product_clean)
    if canon:
        product_clean = canon
    
    # Format according to ReactionT5v2 requirements
    input_text = f"REACTANT:{reactants_clean}REAGENT:{reagents_clean}PRODUCT:{product_clean}"
    
    return input_text


def predict_yield(
    reactants: str,
    reagents: str,
    product: str,
    return_raw: bool = False
) -> float:
    """
    Predicts reaction yield percentage based on reactants, reagents, and product SMILES.
    
    Args:
        reactants: SMILES string of reactants (use . to separate multiple)
        reagents: SMILES or text description of reagents/conditions (can be empty)
        product: SMILES string of the product molecule
        return_raw: If True, returns raw model output before clamping
    
    Returns:
        Predicted yield percentage (0-100)
    """
    # Prepare and canonicalize input
    input_text = prepare_reaction_input(reactants, reagents, product)
    print(f"[DEBUG] Input to model: {input_text}")
    
    # Tokenize
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    # Predict yield
    with torch.no_grad():
        output = model(inputs)
        yield_value = output.item()
    
    print(f"[DEBUG] Raw model output: {yield_value:.2f}%")
    
    if return_raw:
        return round(yield_value, 2)
    
    # Clamp values to [0, 100]
    yield_value = max(0.0, min(100.0, yield_value))
    return round(yield_value, 2)


def predict_yield_batch(reactions: list[Dict[str, str]]) -> list[float]:
    """
    Predict yields for multiple reactions in batch.
    
    Args:
        reactions: List of dicts with keys 'reactants', 'reagents', 'product'
    
    Returns:
        List of predicted yield percentages
    """
    # Prepare all inputs
    input_texts = []
    for rxn in reactions:
        input_text = prepare_reaction_input(
            rxn.get('reactants', ''),
            rxn.get('reagents', ''),
            rxn.get('product', '')
        )
        input_texts.append(input_text)
    
    print(f"[DEBUG] Processing {len(input_texts)} reactions in batch")
    
    # Tokenize all inputs
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    # Predict yields
    with torch.no_grad():
        outputs = model(inputs)
        yield_values = outputs.squeeze().cpu().numpy()
    
    # Handle single prediction case
    if len(reactions) == 1:
        yield_values = [yield_values.item()]
    else:
        yield_values = yield_values.tolist()
    
    # Clamp and round
    results = [round(max(0.0, min(100.0, y)), 2) for y in yield_values]
    
    return results


def validate_reaction_components(reactants: str, reagents: str, product: str) -> bool:
    """
    Validate that all reaction components are valid SMILES.
    
    Args:
        reactants: Reactants SMILES
        reagents: Reagents SMILES (can be empty or text)
        product: Product SMILES
    
    Returns:
        True if all components are valid
    """
    try:
        # Check reactants
        if '.' in reactants:
            for part in reactants.split('.'):
                if not canonicalize_smiles(part):
                    return False
        else:
            if not canonicalize_smiles(reactants):
                return False
        
        # Check product
        if not canonicalize_smiles(product):
            return False
        
        # Check reagents if they look like SMILES
        if reagents and not any(c.isalpha() and c.islower() for c in reagents):
            if '.' in reagents:
                for part in reagents.split('.'):
                    if part.strip() and not canonicalize_smiles(part):
                        return False
            else:
                if not canonicalize_smiles(reagents):
                    return False
        
        return True
    except:
        return False


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Test 1: Simple Nucleophilic Substitution")
    print("="*60)
    reactants1 = "CCBr.C#N"
    product1 = "CCC#N"
    yield1 = predict_yield(reactants1, "", product1)
    print(f"Reactants: {reactants1}")
    print(f"Product: {product1}")
    print(f"Predicted yield: {yield1}%")
    print(f"Valid components: {validate_reaction_components(reactants1, '', product1)}")
    print()
    
    print("="*60)
    print("Test 2: Ester Formation with Catalyst")
    print("="*60)
    reactants2 = "CC(=O)O.CCO"
    reagents2 = "H2SO4"  # Text reagent
    product2 = "CC(=O)OCC"
    yield2 = predict_yield(reactants2, reagents2, product2)
    print(f"Reactants: {reactants2}")
    print(f"Reagents: {reagents2}")
    print(f"Product: {product2}")
    print(f"Predicted yield: {yield2}%")
    print(f"Valid components: {validate_reaction_components(reactants2, reagents2, product2)}")
    print()
    
    print("="*60)
    print("Test 3: Amide Formation")
    print("="*60)
    reactants3 = "CC(=O)Cl.CN"
    reagents3 = ""
    product3 = "CC(=O)NC"
    yield3 = predict_yield(reactants3, reagents3, product3)
    print(f"Reactants: {reactants3}")
    print(f"Product: {product3}")
    print(f"Predicted yield: {yield3}%")
    print(f"Valid components: {validate_reaction_components(reactants3, reagents3, product3)}")
    print()
    
    print("="*60)
    print("Test 4: Batch Yield Prediction")
    print("="*60)
    batch_reactions = [
        {"reactants": "CCBr.C#N", "reagents": "", "product": "CCC#N"},
        {"reactants": "CC(=O)O.CCO", "reagents": "H2SO4", "product": "CC(=O)OCC"},
        {"reactants": "CC(=O)Cl.CN", "reagents": "", "product": "CC(=O)NC"},
    ]
    
    batch_yields = predict_yield_batch(batch_reactions)
    for i, (rxn, y) in enumerate(zip(batch_reactions, batch_yields), 1):
        print(f"{i}. {rxn['reactants']} -> {rxn['product']}")
        print(f"   Predicted yield: {y}%")
    print()
    
    print("="*60)
    print("Test 5: Invalid Reaction Components")
    print("="*60)
    invalid_reactants = "INVALID_SMILES"
    invalid_product = "CCC#N"
    print(f"Valid: {validate_reaction_components(invalid_reactants, '', invalid_product)}")
    print("(Should be False - invalid reactants)")