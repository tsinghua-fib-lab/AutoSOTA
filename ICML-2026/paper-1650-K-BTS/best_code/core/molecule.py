from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class Molecule:
    def __init__(self, raw_smiles, score=None):
        # RDKit Mol
        self.mol = self._prepare_mol(raw_smiles)

        if self.mol:
            # isomericSmiles=True
            self.smiles = Chem.MolToSmiles(self.mol, isomericSmiles=True)
            self.score = score
            # Cache Scaffold SMILES
            self._scaffold_smiles = self._calculate_scaffold()
        else:
            self.smiles = raw_smiles
            self.score = score
            self._scaffold_smiles = None

    def _prepare_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                # Sanitization
                Chem.SanitizeMol(mol)
                return mol
            except:
                return None
        return None

    def _calculate_scaffold(self):
        if not self.mol: return None
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(self.mol)
            # Scaffold  Canonicalize
            return Chem.MolToSmiles(scaf, canonical=True)
        except:
            return None

    @property
    def scaffold_smiles(self):
        return self._scaffold_smiles

    @property
    def scaffold_mol(self):
        return Chem.MolFromSmiles(self._scaffold_smiles) if self._scaffold_smiles else None