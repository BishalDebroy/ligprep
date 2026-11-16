"""
charges.py - Partial charges: Gasteiger, Espaloma or none.
"""

import importlib.util
import logging
import math
import os
import urllib.request
from pathlib import Path
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("ligprep")

CHARGE_MODELS = ("gasteiger", "espaloma", "none")
CHARGE_PROPERTY = "atom.dprop.PartialCharge"

ESPALOMA_MODEL_URL = (
    "https://github.com/choderalab/espaloma_charge/releases/download/v0.0.7/model.pt"
)
ESPALOMA_MODEL_ENV = "LIGPREP_ESPALOMA_MODEL"


class ChargeModelError(RuntimeError):
    """Charge model unavailable, or charge assignment failed."""


def normalize_charge_model(name: str) -> str:
    value = str(name).strip().lower()
    if value not in CHARGE_MODELS:
        raise ValueError(f"Unsupported charge model '{name}'. Choose from: {', '.join(CHARGE_MODELS)}")
    return value


# ---------------------------------------------------------------------------
# Availability (called once, in the main process, before any work starts)
# ---------------------------------------------------------------------------
def espaloma_model_path() -> Path:
    override = os.environ.get(ESPALOMA_MODEL_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "ligprep" / "espaloma_model.pt"


def check_charge_model_available(model: str) -> None:
    """Fail fast, with an actionable message, if a charge model cannot be used.

    Heavy libraries are only located (not imported) here, so the main process
    stays light before worker processes are forked.
    """
    model = normalize_charge_model(model)
    if model != "espaloma":
        return
    missing = [m for m in ("torch", "dgl", "espaloma_charge") if importlib.util.find_spec(m) is None]
    if missing:
        raise ChargeModelError(
            "Espaloma charges need these Python packages: "
            + ", ".join(missing)
            + ". Install them, e.g.  pip install espaloma_charge dgl torch"
        )
    path = espaloma_model_path()
    if path.exists():
        return
    logger.info(f"Downloading Espaloma charge model to {path} (one-time download)")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(ESPALOMA_MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as e:
        raise ChargeModelError(
            f"Could not download the Espaloma model ({e}). Download {ESPALOMA_MODEL_URL} "
            f"manually and point the {ESPALOMA_MODEL_ENV} environment variable to the file."
        ) from e


# ---------------------------------------------------------------------------
# Charge computation
# ---------------------------------------------------------------------------
def rectify_charges(charges: List[float], net_charge: int, decimals: int = 3) -> List[float]:
    """Round to ``decimals`` and spread the rounding residue so the sum equals ``net_charge``."""
    scale = 10 ** decimals
    units = [int(round(q * scale)) for q in charges]
    surplus = int(round(net_charge * scale)) - sum(units)
    if surplus and units:
        order = sorted(range(len(units)), key=lambda i: -abs(charges[i]))
        step = 1 if surplus > 0 else -1
        for k in range(abs(surplus)):
            units[order[k % len(order)]] += step
    return [u / scale for u in units]


def _gasteiger(mol_h: Chem.Mol) -> List[float]:
    work = Chem.Mol(mol_h)
    AllChem.ComputeGasteigerCharges(work)
    charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in work.GetAtoms()]
    if not all(math.isfinite(q) for q in charges):
        raise ChargeModelError("Gasteiger charges are not finite for this molecule")
    return charges


_ESPALOMA = {}


def _espaloma(mol_h: Chem.Mol) -> List[float]:
    """Machine-learned AM1-BCC charges. The model is loaded once per worker process."""
    try:
        if "model" not in _ESPALOMA:
            import torch
            torch.set_num_threads(1)
            path = str(espaloma_model_path())
            try:
                model = torch.load(path, weights_only=False)   # model is a pickled module
            except TypeError:                                  # older torch: no such argument
                model = torch.load(path)
            _ESPALOMA["model"] = model
        import torch
        from espaloma_charge.utils import from_rdkit_mol
        graph = from_rdkit_mol(mol_h)
        with torch.no_grad():
            graph = _ESPALOMA["model"](graph)
        return [float(q) for q in graph.ndata["q"].cpu().detach().flatten().numpy()]
    except ChargeModelError:
        raise
    except Exception as e:
        raise ChargeModelError(f"Espaloma charge calculation failed: {e}") from e


def assign_charges(mol_h: Chem.Mol, model: str) -> Chem.Mol:
    """Store partial charges on ``mol_h`` (explicit hydrogens) in place and return it."""
    model = normalize_charge_model(model)
    if model == "none":
        return mol_h
    raw = _gasteiger(mol_h) if model == "gasteiger" else _espaloma(mol_h)
    if len(raw) != mol_h.GetNumAtoms():
        raise ChargeModelError("Charge model returned the wrong number of charges")
    charges = rectify_charges(raw, Chem.GetFormalCharge(mol_h))
    mol_h.SetProp(CHARGE_PROPERTY, " ".join(f"{q + 0.0:.3f}" for q in charges))
    return mol_h