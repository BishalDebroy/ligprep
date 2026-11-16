"""
protonate.py - Major protonation state at a given pH using Dimorphite-DL.

Dimorphite-DL works on SMILES, so this returns a topology (no explicit hydrogens,
no 3D); 3D coordinates are generated afterwards by conformers.py. Only the
top-ranked state is used.
"""

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Tuple

import dimorphite_dl
from rdkit import Chem

logger = logging.getLogger("ligprep")


def protonate(
        mol: Chem.Mol,
        pH: float = 7.4,
        min_ph: Optional[float] = None,
        max_ph: Optional[float] = None,
        max_states: int = 1,
) -> Optional[Chem.Mol]:
    """
    Return the top-ranked protonation state at the specified pH (or pH range),
    or None on error. If Dimorphite-DL proposes no change the input topology is
    returned unchanged.
    """
    if min_ph is None:
        min_ph = pH
    if max_ph is None:
        max_ph = pH

    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        variants = dimorphite_dl.protonate_smiles(
            smi,
            ph_min=min_ph,
            ph_max=max_ph,
            max_variants=max_states,
        )
    except Exception as e:
        logger.error(f"Dimorphite-DL error: {e}")
        return None

    if not variants:
        return Chem.RemoveHs(mol)

    best_mol = Chem.MolFromSmiles(variants[0])
    if best_mol is None:
        logger.warning(f"Could not parse protonated SMILES: {variants[0]}")
        return None
    return best_mol


# Dimorphite-DL 2.0.x leaves aliphatic amines neutral at pH 7.4 and ionises phenols and
# imidazole/triazole N-H; 2.1.0 fixes this.
MIN_RECOMMENDED_DIMORPHITE = (2, 1, 0)


def _version_tuple(text: str) -> Tuple[int, int, int]:
    parts = []
    for piece in text.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def dimorphite_version_warning() -> Optional[str]:
    """Warning text if the installed Dimorphite-DL is older than recommended, else None."""
    try:
        installed = version("dimorphite-dl")
    except PackageNotFoundError:
        return None
    if _version_tuple(installed) >= MIN_RECOMMENDED_DIMORPHITE:
        return None
    return (f"Dimorphite-DL {installed} is installed. Versions before 2.1.0 leave aliphatic amines "
            f"neutral at pH 7.4 and ionise phenols and imidazole/triazole N-H, so protonation states "
            f"will be wrong. Upgrade with:  pip install -U \"dimorphite-dl>=2.1,<3.0\"")
