import torch
from dgllife.model import model_zoo
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer

# ✅ Model configuration (same as training)
N_LAYERS = 2
N_TIMESTEPS = 1
GRAPH_FEAT_SIZE = 200
N_TASKS = 8  # change this if your final classification task differs
DROPOUT = 0.1

MODEL_PATH = "app/models/final_trained_ReactAIvate_model.pth"

# Load featurizers
atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')

n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

# Initialize model structure
model = model_zoo.AttentiveFPPredictor(
    node_feat_size=n_feats,
    edge_feat_size=e_feats,
    num_layers=N_LAYERS,
    num_timesteps=N_TIMESTEPS,
    graph_feat_size=GRAPH_FEAT_SIZE,
    n_tasks=N_TASKS,
    dropout=DROPOUT
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()  # inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ ReactAIvate model loaded successfully!")
