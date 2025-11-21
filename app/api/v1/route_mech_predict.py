from fastapi import APIRouter, HTTPException
from app.schemas.mech_predict_schema import ReactAIvateInput, ReactAIvateOutput
from app.core.model_loader import model, atom_featurizer, bond_featurizer, device
from dgllife.utils import smiles_to_bigraph
import torch
import torch.nn.functional as F

router = APIRouter(prefix="/reactaivate", tags=["ReactAIvate"])

@router.post("/predict", response_model=ReactAIvateOutput)
def predict(data: ReactAIvateInput):
    try:
        # 1️⃣ Convert SMILES → molecular graph
        mol_graph = smiles_to_bigraph(
            data.smiles,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer
        )

        mol_graph = mol_graph.to(device)

        # 2️⃣ Extract node & edge features
        node_feats = mol_graph.ndata['hv'].to(device)
        edge_feats = mol_graph.edata['he'].to(device)

        # 3️⃣ Forward pass
        with torch.no_grad():
            logits = model(mol_graph, node_feats, edge_feats)
            probs = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
            pred_class = int(torch.argmax(logits, dim=1).cpu().item())

        return ReactAIvateOutput(
            predicted_class=pred_class,
            probabilities=probs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
