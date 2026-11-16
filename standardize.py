"""
standardize.py - standardize SMILES: remove salts, normalize, remove duplicate fragments.

Returns a topology only. 3D coordinates are generated later (conformers.py), once
the protonation state is known, so each molecule is embedded exactly once.
"""

import logging
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger("ligprep")


def standardize_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], str]:
    """
    Steps:
    - Parse the SMILES
    - Remove salts, keep the largest fragment
    - Normalize functional groups, aromatize, sanitize
    - Remove duplicate fragments (if any)

    Returns: (mol or None, canonical_smiles)
    """

    # -------------Step 1: Initial molecule from SMILES---------------------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, ""

    # ------------Step 2: Salt removal & keep largest fragment--------------
    chooser = rdMolStandardize.LargestFragmentChooser()
    mol = chooser.choose(mol)
    if mol is None:
        return None, ""

    # ------------Step 3: Normalize functional groups, aromatize, sanitize--
    normalizer = rdMolStandardize.Normalizer()
    mol = normalizer.normalize(mol)
    # Remove duplicate fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        seen = set()
        unique_frags = []
        for f in frags:
            cansmi = Chem.MolToSmiles(f, canonical=True)
            if cansmi not in seen:
                seen.add(cansmi)
                unique_frags.append(f)
        if unique_frags:
            mol = unique_frags[0]
            for f in unique_frags[1:]:
                mol = Chem.CombineMols(mol, f)
    # Ensure aromaticity is perceived after all modifications
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    return mol, Chem.MolToSmiles(mol, canonical=True)
