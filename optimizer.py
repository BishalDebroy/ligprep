"""
optimizer.py - Force-field minimisation (UFF, MMFF94, MMFF94s) and energies.
"""

import logging
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("ligprep")

SUPPORTED_FORCE_FIELDS = ("UFF", "MMFF94", "MMFF94s")
MAX_FF_ITERATIONS = 2000


def normalize_forcefield_name(name: str) -> str:
    """Return the canonical spelling ('UFF', 'MMFF94', 'MMFF94s') of a force-field name."""
    lookup = {n.lower(): n for n in SUPPORTED_FORCE_FIELDS}
    try:
        return lookup[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported force field '{name}'. Choose from: {', '.join(SUPPORTED_FORCE_FIELDS)}"
        ) from None


def require_explicit_hydrogens(mol: Chem.Mol) -> None:
    """Raise ValueError if the molecule still carries implicit hydrogens.

    Force-field energies of a hydrogen-less structure are meaningless, so this
    is checked explicitly instead of letting the optimiser run silently.
    """
    if any(atom.GetNumImplicitHs() > 0 for atom in mol.GetAtoms()):
        raise ValueError("Force-field optimisation requires explicit hydrogens (use Chem.AddHs).")


def resolve_forcefield(mol: Chem.Mol, requested: str) -> str:
    """Return the force field that will actually be used for this molecule."""
    requested = normalize_forcefield_name(requested)
    if requested in ("MMFF94", "MMFF94s"):
        if AllChem.MMFFHasAllMoleculeParams(mol):
            return requested
        logger.debug(f"{requested} parameters unavailable for this molecule; falling back to UFF")
    if AllChem.UFFHasAllMoleculeParams(mol):
        return "UFF"
    raise ValueError("No force-field parameters (MMFF or UFF) available for this molecule.")


def optimize_conformers(
        mol: Chem.Mol,
        requested: str = "MMFF94",
        max_iters: int = MAX_FF_ITERATIONS,
) -> Tuple[List[float], str]:
    """Minimise every conformer of ``mol`` in place.

    Returns (energies in kcal/mol, in the order of mol.GetConformers(), force field used).
    """
    require_explicit_hydrogens(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers to optimise.")
    used = resolve_forcefield(mol, requested)
    if used == "UFF":
        results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=max_iters)
    else:
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, maxIters=max_iters, mmffVariant=used
        )
    energies = [float(energy) for _, energy in results]
    if any(status < 0 for status, _ in results):
        raise ValueError(f"{used} setup failed for this molecule.")
    return energies, used


def minimize_conformer(mol: Chem.Mol, force_field: str, conf_id: int,
                       max_iters: int = MAX_FF_ITERATIONS) -> float:
    """Minimise one conformer in place with an already resolved force field; returns kcal/mol."""
    if force_field == "UFF":
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    else:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if ff is None:
        raise ValueError(f"{force_field} setup failed for conformer {conf_id}")
    ff.Minimize(maxIts=max_iters)
    return float(ff.CalcEnergy())


def optimize_and_energy(
        mol: Chem.Mol,
        force_field: str = "MMFF94",
) -> Tuple[Optional[Chem.Mol], Optional[float], str]:
    """Optimise a molecule (explicit Hs, >=1 conformer) and return a copy.

    Returns (optimised mol, energy of the first conformer in kcal/mol,
    force field used); (None, None, "") if optimisation is impossible.
    """
    work = Chem.Mol(mol)
    try:
        energies, used = optimize_conformers(work, force_field)
    except Exception as e:
        logger.warning(f"Optimisation failed: {e}")
        return None, None, ""
    return work, energies[0], used
