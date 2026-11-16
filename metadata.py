"""
metadata.py - Build SD tags for first and second SDF entries.
"""

from typing import Dict, Optional
from importlib.metadata import version, PackageNotFoundError
from rdkit import Chem

try:
    _DIMORPHITE_VERSION = version("dimorphite-dl")
except PackageNotFoundError:
    _DIMORPHITE_VERSION = "unknown"

def first_sdf_tags(
        compound_name: str,
        original_smi: str,
        canonical_smi: str,
        force_field: str,
        ph: float,
        uuid: str,
        generation_date: str,
) -> Dict[str, str]:
    """Return dictionary of SD tag name → value for first SDF."""
    return {
        "Compound_Name": compound_name,
        "Original_SMILES": original_smi,
        "Canonical_SMILES": canonical_smi,
        "Generation_Date": generation_date,
        "Preparation_Software": "RDKit;Dimorphite-DL",
        "RDKit_version": Chem.rdBase.rdkitVersion,
        "Dimorphite-DL_version": _DIMORPHITE_VERSION,
        "Force_field": force_field,
        "pH": str(ph),
        "Processing_Step": "standardization+protonation",
        "UUID": uuid
    }

def second_sdf_tags(
        parent_tags: Dict[str, str],
        tautomer_name: str,
        tautomer_number: int,
        is_canonical: bool,
        energy: Optional[float],
        energy_unit: str,
        opt_force_field: str,
        *,
        tautomer_smiles: str = "",
        relative_energy: Optional[float] = None,   # vs lowest ring state of the SAME tautomer
        ring_conformer: str = "NA",
        ring_state_number: int = 1,
        charge_model: str = "none",
) -> Dict[str, str]:
    """Return tags for a tautomer entry, retaining all first SDF fields."""
    new_tag = {k: str(v) for k, v in parent_tags.items()}  # copy all original fields
    new_tag.update({
        "Processing_Step": "tautomer_enumeration+3D_optimization",
        "Tautomer_Name": tautomer_name,
        "Tautomer_Number": str(tautomer_number),
        "Tautomer_SMILES": tautomer_smiles,
        "Parent_compound": parent_tags["Compound_Name"],
        "Is_Canonical_Tautomer": "1" if is_canonical else "0",
        "Energy": f"{energy:.4f}" if energy is not None else "N/A",
        "Ring_State_Relative_Energy": f"{relative_energy:.4f}" if relative_energy is not None else "N/A",
        "Energy_Unit": energy_unit,
        "Optimization_ForceField": opt_force_field,
        "Ring_Conformer": ring_conformer,
        "Ring_State_Number": str(ring_state_number),
        "Charge_Model": charge_model,
    })
    return new_tag
