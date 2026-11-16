"""
tautomer.py - pH-dependent, low-energy tautomers with the canonical one flagged.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from conformers import ConformerError, ConformerState, generate_states
from protonate import protonate

logger = logging.getLogger("ligprep")

MAX_TAUTOMERS_ENUMERATED = 200     # RDKit enumeration cap
MAX_TAUTOMER_CANDIDATES = 50       # best RDKit-scored tautomers examined further

# The RDKit canonical tautomer is flagged in the output whenever it is among the
# low-energy tautomers. With False (default) it is dropped when the SMARTS rules
# find it high in energy; with True it is always written, flagged, even then.
ALWAYS_KEEP_CANONICAL = False

# SMARTS stability rules, applied in this order of priority.
#   "max": prefer tautomers with the largest number of matches
#   "min": prefer tautomers with the smallest number of matches
# Carbonyls on N or O rank above aromaticity on purpose: RDKit (and a naive
# "most aromatic" rule) turns imides, hydantoins, thiazolidinediones, barbiturates
# and anhydrides into aromatic hydroxy-azoles/-furans that do not exist in practice,
# while phenols still keep their aromatic ring (their keto forms have no C=O on N/O).
STABILITY_RULES: List[Tuple[str, str, str]] = [
    ("max", "carbonyl on N or O (amide, imide, urea, ester, lactone, acid)",
     "[OX1,SX1]=[#6X3][#7X3,#7X2-,#8X2]"),
    ("max", "aromatic atom",         "[a]"),
    ("min", "enol",                  "[#6X3]=[#6X3][OX2H1]"),
    ("min", "imidic acid",           "[#6X3](=[NX2])[OX2H1]"),
    # enamines, except vinylogous amides (N-C=C-C=O), which are stabilised by conjugation
    ("min", "enamine",
     "[NX3;+0;!$(N-[#6]=[O,S]);!$(N-[#6]=[#6]-[#6]=[O,S])]-[#6X3;!a]=[#6X3;!a]"),
    ("min", "exocyclic ring imine",  "[NX2;+0;!a]=[c]"),
]
_RULE_PATTERNS = [(kind, name, Chem.MolFromSmarts(sma)) for kind, name, sma in STABILITY_RULES]
assert all(p is not None for _, _, p in _RULE_PATTERNS), "invalid SMARTS in STABILITY_RULES"

_ENUMERATORS: Dict[str, rdMolStandardize.TautomerEnumerator] = {}


def _get_enumerator(transforms_file: Optional[str] = None) -> rdMolStandardize.TautomerEnumerator:
    """Cached enumerator (built once per process). ``transforms_file`` = custom SMARTS rules."""
    key = transforms_file or ""
    if key not in _ENUMERATORS:
        if transforms_file:
            params = rdMolStandardize.CleanupParameters()
            params.tautomerTransformsFile = str(transforms_file)
            params.maxTautomers = MAX_TAUTOMERS_ENUMERATED
            params.tautomerRemoveSp3Stereo = False
            params.tautomerRemoveBondStereo = False
            enumerator = rdMolStandardize.TautomerEnumerator(params)
        else:
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.SetMaxTautomers(MAX_TAUTOMERS_ENUMERATED)
            enumerator.SetRemoveSp3Stereo(False)
            enumerator.SetRemoveBondStereo(False)
        _ENUMERATORS[key] = enumerator
    return _ENUMERATORS[key]


def enumerate_tautomers(
        mol: Chem.Mol,
        transforms_file: Optional[str] = None,
        max_candidates: int = MAX_TAUTOMER_CANDIDATES,
) -> List[Tuple[Chem.Mol, bool]]:
    """
    Return [(tautomer_mol, is_canonical)], canonical first, then by RDKit tautomer
    score (best first), capped at ``max_candidates``. Topologies only (no Hs, no 3D).
    """
    mol = Chem.RemoveHs(mol)
    try:
        enumerator = _get_enumerator(transforms_file)
        result = enumerator.Enumerate(mol)
        canonical = enumerator.Canonicalize(mol)
    except Exception as e:
        logger.warning(f"Tautomer enumeration failed ({e}); using the input structure only.")
        return [(mol, True)]

    canon_smi = Chem.MolToSmiles(canonical)
    entries: Dict[str, Tuple[Chem.Mol, bool, int]] = {}
    for taut in (list(result.tautomers) if result else []):
        smi = Chem.MolToSmiles(taut)
        if smi not in entries:
            entries[smi] = (taut, smi == canon_smi,
                            rdMolStandardize.TautomerEnumerator.ScoreTautomer(taut))
    if canon_smi not in entries:
        entries[canon_smi] = (canonical, True,
                              rdMolStandardize.TautomerEnumerator.ScoreTautomer(canonical))

    kept = _drop_stereo_lossy(mol, list(entries.values()))
    ordered = sorted(kept, key=lambda e: (not e[1], -e[2]))
    return [(m, is_canon) for m, is_canon, _ in ordered[:max_candidates]]


def _unassigned_stereocentres(mol: Chem.Mol) -> int:
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return sum(1 for _, label in centres if label == "?")


def _drop_stereo_lossy(parent: Chem.Mol, entries: List[Tuple[Chem.Mol, bool, int]]):
    """Remove tautomers that lost (or newly expose) undefined stereocentres.

    Passing through a planar intermediate (e.g. an enol) erases a stereocentre, so
    RDKit can return, next to the real tautomer, a copy of the same constitution
    with the centre undefined. Embedding that copy would assign a random
    configuration, i.e. silently create a wrong stereoisomer. Two rules:
    * among tautomers of one constitution only the most fully specified are kept
      (a dropped canonical flag moves to the survivor);
    * other tautomers may not have more undefined stereocentres than the parent.
    """
    info = [(m, canon, score, _unassigned_stereocentres(m), Chem.MolToSmiles(m, isomericSmiles=False))
            for m, canon, score in entries]
    best_in_group: Dict[str, int] = {}
    for _, _, _, n_un, flat in info:
        best_in_group[flat] = min(best_in_group.get(flat, n_un), n_un)
    survivors = {}
    for m, canon, score, n_un, flat in info:
        if n_un == best_in_group[flat]:
            survivors.setdefault(flat, []).append([m, canon, score, n_un])
    for m, canon, score, n_un, flat in info:       # hand the canonical flag to a survivor
        if canon and n_un != best_in_group[flat]:
            survivors[flat][0][1] = True
    parent_un = _unassigned_stereocentres(parent)
    kept = []
    for group in survivors.values():
        for m, canon, score, n_un in group:
            if canon or n_un <= parent_un:
                kept.append((m, canon, score))
    return kept


# ---------------------------------------------------------------------------
# Candidates, pH dependence
# ---------------------------------------------------------------------------
def stability_key(mol: Chem.Mol) -> Tuple[int, ...]:
    """Sort key from the SMARTS stability rules: smaller is more stable."""
    key = []
    for kind, _, pattern in _RULE_PATTERNS:
        n = len(mol.GetSubstructMatches(pattern))
        key.append(-n if kind == "max" else n)
    return tuple(key)


@dataclass
class TautomerCandidate:
    mol: Chem.Mol            # pH-adjusted topology
    smiles: str
    is_canonical: bool
    rank: int = 0            # position in the RDKit-scored enumeration (0 = best)
    key: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def net_charge(self) -> int:
        return Chem.GetFormalCharge(self.mol)


def adjust_tautomers_to_ph(
        tautomers: List[Tuple[Chem.Mol, bool]],
        pH: float = 7.4,
        min_ph: Optional[float] = None,
        max_ph: Optional[float] = None,
        max_states: int = 1,
) -> List[TautomerCandidate]:
    """Re-protonate every tautomer at the given pH (top-ranked state) and merge duplicates.

    A tautomer whose net charge would change is not a microstate of the species
    that dominates at this pH: it is dropped, except the canonical tautomer,
    which is kept exactly as enumerated.
    """
    unique: Dict[str, TautomerCandidate] = {}
    for rank, (tmol, is_canon) in enumerate(tautomers):
        adjusted = protonate(tmol, pH, min_ph, max_ph, max_states)
        if adjusted is None:
            logger.debug("pH adjustment failed for a tautomer; keeping it as enumerated")
            adjusted = tmol
        elif Chem.GetFormalCharge(adjusted) != Chem.GetFormalCharge(tmol):
            if not is_canon:
                logger.debug(f"Dropping tautomer {Chem.MolToSmiles(tmol)}: it would ionise "
                             f"differently at this pH ({Chem.MolToSmiles(adjusted)})")
                continue
            adjusted = tmol
        smi = Chem.MolToSmiles(adjusted)
        if smi in unique:
            unique[smi].is_canonical = unique[smi].is_canonical or is_canon
        else:
            unique[smi] = TautomerCandidate(adjusted, smi, is_canon, rank, stability_key(adjusted))
    return list(unique.values())


# ---------------------------------------------------------------------------
# Low-energy selection (SMARTS rules)
# ---------------------------------------------------------------------------
def filter_low_energy(
        candidates: List[TautomerCandidate],
        always_keep_canonical: bool = None,
) -> List[TautomerCandidate]:
    """Keep the tautomers with the best SMARTS stability key (ties are all kept).

    The RDKit canonical tautomer is additionally kept only if ``always_keep_canonical``
    (default: ALWAYS_KEEP_CANONICAL). Result is ordered most stable first; among equally
    stable tautomers the canonical one comes first.
    """
    if always_keep_canonical is None:
        always_keep_canonical = ALWAYS_KEEP_CANONICAL
    if not candidates:
        return []
    best = min(c.key for c in candidates)
    kept = [c for c in candidates if c.key == best or (always_keep_canonical and c.is_canonical)]
    kept.sort(key=lambda c: (c.key, not c.is_canonical, c.rank))
    return kept


@dataclass
class ScoredTautomer:
    candidate: TautomerCandidate
    states: List[ConformerState]     # ring states, lowest energy first


def prepare_tautomer_states(
        mol: Chem.Mol,
        force_field: str = "MMFF94",
        pH: float = 7.4,
        min_ph: Optional[float] = None,
        max_ph: Optional[float] = None,
        max_states: int = 1,
        transforms_file: Optional[str] = None,
) -> List[ScoredTautomer]:
    """Full tautomer stage for one protonated molecule (see module docstring)."""
    tautomers = enumerate_tautomers(mol, transforms_file)
    candidates = filter_low_energy(adjust_tautomers_to_ph(tautomers, pH, min_ph, max_ph, max_states))
    result = []
    for cand in candidates:
        try:
            states = generate_states(cand.mol, force_field, enumerate_ring_states=True)
        except ConformerError as e:
            logger.debug(f"Skipping tautomer {cand.smiles}: {e}")
            continue
        result.append(ScoredTautomer(cand, states))
    return result
