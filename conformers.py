"""
conformers.py - 3D generation with chair/boat handling of saturated six-membered rings.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

from optimizer import minimize_conformer, optimize_conformers

logger = logging.getLogger("ligprep")

EMBED_SEED = 42                # reproducible embedding
CONFS_SINGLE_RING = 12         # ensemble size when there is one saturated six-membered ring
CONFS_PER_RING_MULTI = 20      # per ring when there are several: coupled rings need denser sampling
MAX_ENSEMBLE_CONFS = 60        # hard cap on the ensemble size
RING_ENERGY_WINDOW = 0.5       # kcal/mol: ring states kept relative to the lowest
MAX_RING_STATES = 4            # at most this many ring states per structure
CHAIR_MIN_TORSION = 35.0       # degrees: every ring torsion must exceed this in a chair
RING_STATE_RMSD = 0.25         # Angstrom: below this two conformers are the same state


class ConformerError(RuntimeError):
    """Raised when no 3D structure can be generated."""


@dataclass
class ConformerState:
    """One 3D structure: explicit hydrogens, exactly one conformer."""
    mol: Chem.Mol
    energy: float            # kcal/mol, in the force field named below
    force_field: str         # force field actually used
    ring_conformer: str      # 'NA', 'chair', 'mixed' or 'non-chair'


# ---------------------------------------------------------------------------
# Ring perception and classification
# ---------------------------------------------------------------------------
def saturated_six_rings(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Six-membered rings whose atoms are all non-aromatic and joined by single bonds."""
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        bonds = [mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % 6]) for k in range(6)]
        if all(b is not None and b.GetBondType() == Chem.BondType.SINGLE for b in bonds):
            rings.append(tuple(ring))
    return rings


def is_chair(conf: Chem.Conformer, ring: Sequence[int], min_abs: float = CHAIR_MIN_TORSION) -> bool:
    """True if the six endocyclic torsions alternate in sign and all exceed ``min_abs`` degrees."""
    tors = [
        rdMolTransforms.GetDihedralDeg(
            conf, ring[i], ring[(i + 1) % 6], ring[(i + 2) % 6], ring[(i + 3) % 6]
        )
        for i in range(6)
    ]
    return all(abs(t) >= min_abs for t in tors) and all(
        tors[i] * tors[(i + 1) % 6] < 0 for i in range(6)
    )


def _ring_system_subset(mol_h: Chem.Mol, rings, conf_ids: Sequence[int]) -> Chem.Mol:
    """Small coloured copy (ring atoms + heavy first neighbours) used to compare ring states.

    Atoms are coloured by their molecular symmetry class (chirality ignored, so
    differently labelled copies of the same physical structure still coincide), so the
    symmetry-aware RMSD may only swap atoms that are constitutionally equivalent.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False, includeChirality=False))
    keep = set()
    for ring in rings:
        for i in ring:
            keep.add(i)
            for nbr in mol_h.GetAtomWithIdx(i).GetNeighbors():
                if nbr.GetAtomicNum() > 1:
                    keep.add(nbr.GetIdx())
    keep = sorted(keep)
    index = {a: i for i, a in enumerate(keep)}
    classes = sorted({ranks[a] for a in keep})
    colour = {c: 20 + i for i, c in enumerate(classes)}   # fake atomic numbers = colours

    rw = Chem.RWMol()
    for a in keep:
        rw.AddAtom(Chem.Atom(colour[ranks[a]]))
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in index and j in index:
            rw.AddBond(index[i], index[j], Chem.BondType.SINGLE)
    sub = rw.GetMol()
    for cid in conf_ids:
        src = mol_h.GetConformer(cid)
        conf = Chem.Conformer(len(keep))
        for a in keep:
            conf.SetAtomPosition(index[a], src.GetAtomPosition(a))
        conf.SetId(cid)
        sub.AddConformer(conf, assignId=False)
    return sub


def _symmetric_rms(sub: Chem.Mol, cid_a: int, cid_b: int) -> float:
    return rdMolAlign.GetBestRMS(Chem.Mol(sub), sub, prbId=cid_a, refId=cid_b, maxMatches=2000)


# ---------------------------------------------------------------------------
# Embedding and state generation
# ---------------------------------------------------------------------------
def _ensemble_size(n_rings: int) -> int:
    if n_rings == 0:
        return 1
    if n_rings == 1:
        return CONFS_SINGLE_RING
    return min(MAX_ENSEMBLE_CONFS, CONFS_PER_RING_MULTI * n_rings)


def _embed(mol_h: Chem.Mol, n_confs: int, seed: int) -> List[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = -1.0
    ids = list(AllChem.EmbedMultipleConfs(mol_h, n_confs, params))
    if not ids:                       # second attempt, more forgiving
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol_h, n_confs, params))
    return ids


def _single_conformer_copy(mol_h: Chem.Mol, cid: int) -> Chem.Mol:
    new = Chem.Mol(mol_h)
    new.RemoveAllConformers()
    conf = Chem.Conformer(mol_h.GetConformer(cid))
    conf.SetId(0)
    new.AddConformer(conf, assignId=False)
    return new


def _select_ring_states(mol_h, cids, energies, flags, rings, required, enumerate_ring_states):
    """Pick the distinct chair states: indices into ``cids``, lowest energy first."""
    def n_chairs(i: int) -> int:
        return sum(flags[i][k] for k in required)

    top = max(n_chairs(i) for i in range(len(cids)))
    candidates = sorted((i for i in range(len(cids)) if n_chairs(i) == top), key=lambda i: energies[i])
    sub = _ring_system_subset(mol_h, rings, [cids[i] for i in candidates])
    e_ref = energies[candidates[0]]
    accepted: List[int] = []
    for i in candidates:
        if energies[i] - e_ref > RING_ENERGY_WINDOW or len(accepted) >= MAX_RING_STATES:
            break
        if all(_symmetric_rms(sub, cids[i], cids[j]) > RING_STATE_RMSD for j in accepted):
            accepted.append(i)
    if not enumerate_ring_states or not required:
        accepted = accepted[:1]       # rigid / non-chair-capable rings: one state is enough
    return accepted


def _stereo_signature(mol_h: Chem.Mol, cid: int) -> Tuple:
    """Per-atom / per-bond stereo configuration actually present in a conformer's geometry.

    Atom and bond indices are those of ``mol_h``, so two conformers have the same
    signature only if they realise the *same labelled* stereoisomer (a canonical
    SMILES comparison would also accept the relabelled mirror form of a meso compound).
    """
    tmp = Chem.Mol(mol_h)
    tmp.RemoveAllConformers()
    tmp.AddConformer(Chem.Conformer(mol_h.GetConformer(cid)), assignId=True)
    Chem.AssignStereochemistryFrom3D(tmp)
    return (tuple(int(a.GetChiralTag()) for a in tmp.GetAtoms()),
            tuple(int(b.GetStereo()) for b in tmp.GetBonds()))


def _mirror_partner(mol_h: Chem.Mol, cid: int) -> Optional[Chem.Conformer]:
    """Mirror image of a conformer, relabelled so the molecule's stereochemistry is unchanged.

    Reflecting a conformer inverts every stereocentre; a symmetry of the molecular
    graph that swaps the affected centres puts them back. If no such symmetry
    exists (a chiral molecule) there is no mirror partner and None is returned.
    """
    heavy = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
    heavy_mol = Chem.RemoveHs(mol_h)
    pos = np.array(mol_h.GetConformer(cid).GetPositions())
    reflected = pos.copy()
    reflected[:, 0] *= -1.0
    reference = _stereo_signature(mol_h, cid)

    # hydrogens grouped by the heavy atom they sit on
    h_of = {i: sorted(n.GetIdx() for n in mol_h.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1)
            for i in heavy}
    for match in heavy_mol.GetSubstructMatches(heavy_mol, uniquify=False, useChirality=False, maxMatches=500):
        mapping = {}                                   # old atom index -> index it moves to
        for i, j in zip(heavy, (heavy[k] for k in match)):
            mapping[i] = j
            if len(h_of[i]) != len(h_of[j]):
                break
            for hi, hj in zip(h_of[i], h_of[j]):
                mapping[hi] = hj
        else:
            if len(mapping) != mol_h.GetNumAtoms():
                continue
            new_pos = np.zeros_like(pos)
            for old, new in mapping.items():
                new_pos[new] = reflected[old]
            conf = Chem.Conformer(mol_h.GetNumAtoms())
            for k in range(mol_h.GetNumAtoms()):
                conf.SetAtomPosition(k, new_pos[k].tolist())
            probe = Chem.Mol(mol_h)
            probe.RemoveAllConformers()
            probe.AddConformer(conf, assignId=True)
            if _stereo_signature(probe, 0) == reference:
                return conf
    return None


def generate_states(
        mol: Chem.Mol,
        force_field: str = "MMFF94",
        enumerate_ring_states: bool = True,
        seed: int = EMBED_SEED,
) -> List[ConformerState]:
    """Build minimised 3D structures (explicit hydrogens) for a molecule.

    ``mol`` is a topology (hydrogens optional). With ``enumerate_ring_states``
    False exactly one state (the lowest-energy chair arrangement) is returned;
    with True, every distinct ring state within RING_ENERGY_WINDOW is returned.
    Raises ConformerError if no 3D structure can be built.
    """
    mol_h = Chem.AddHs(mol)
    rings = saturated_six_rings(mol_h)
    n_confs = _ensemble_size(len(rings))

    if not _embed(mol_h, n_confs, seed):
        raise ConformerError("3D embedding failed")
    cids = [c.GetId() for c in mol_h.GetConformers()]
    try:
        energies, used_ff = optimize_conformers(mol_h, force_field)
    except Exception as e:
        raise ConformerError(f"force-field optimisation failed: {e}") from e

    if not rings:
        best = min(range(len(cids)), key=lambda i: energies[i])
        return [ConformerState(_single_conformer_copy(mol_h, cids[best]),
                               energies[best], used_ff, "NA")]

    flags = [[is_chair(mol_h.GetConformer(cid), ring) for ring in rings] for cid in cids]
    can_chair = [any(flags[i][k] for i in range(len(cids))) for k in range(len(rings))]
    required = [k for k, ok in enumerate(can_chair) if ok]

    accepted = _select_ring_states(mol_h, cids, energies, flags, rings, required, enumerate_ring_states)

    if enumerate_ring_states and required:
        # complete the set with the exact mirror-image partners (see module docstring)
        for i in list(accepted):
            partner = _mirror_partner(mol_h, cids[i])
            if partner is None:
                continue
            new_id = mol_h.AddConformer(partner, assignId=True)
            try:
                energy = minimize_conformer(mol_h, used_ff, new_id)
            except Exception as e:
                logger.debug(f"Mirror partner could not be minimised: {e}")
                mol_h.RemoveConformer(new_id)
                continue
            cids.append(new_id)
            energies.append(energy)
            flags.append([is_chair(mol_h.GetConformer(new_id), ring) for ring in rings])
        accepted = _select_ring_states(mol_h, cids, energies, flags, rings, required, True)

    states = []
    for i in accepted:
        n_ok = sum(flags[i])
        label = "chair" if n_ok == len(rings) else ("mixed" if n_ok else "non-chair")
        states.append(ConformerState(_single_conformer_copy(mol_h, cids[i]),
                                     energies[i], used_ff, label))
    return states