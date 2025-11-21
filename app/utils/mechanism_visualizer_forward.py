from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Chem import Draw

def generate_forward_visual(reactants: str, reagents: str, product: str):
    try:
        rxn_smi = f"{reactants}>>{product}"
        rxn = Chem.ReactionFromSmarts(rxn_smi, useSmiles=True)

        drawer = Draw.ReactionDrawer()
        opts = drawer.GetDrawOptions()
        opts.padding = 0.2
        opts.bondLineWidth = 2
        opts.arrowLineWidth = 2
        opts.dpi = 120

        svg = drawer.DrawReactionToSVG(
            rxn,
            width=700,
            height=300
        )

        return svg

    except Exception as e:
        return f"<svg><!-- error: {e} --></svg>"


def generate_3d_molblock(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    return Chem.MolToMolBlock(mol)
