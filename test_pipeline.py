"""
Usage: python -m pytest test_pipeline.py -v
"""

import importlib.util
import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rdkit import Chem
from rdkit.Chem import AllChem

from charges import (CHARGE_PROPERTY, ChargeModelError, assign_charges,
                     check_charge_model_available, normalize_charge_model, rectify_charges)
from conformers import ConformerError, _mirror_partner, generate_states, is_chair, saturated_six_rings
from optimizer import normalize_forcefield_name, optimize_and_energy, optimize_conformers
import protonate as protonate_module
from protonate import protonate, dimorphite_version_warning
from reader import read_smiles
from standardize import standardize_smiles
import tautomer as tautomer_module
from tautomer import (TautomerCandidate, adjust_tautomers_to_ph, enumerate_tautomers,
                      filter_low_energy, prepare_tautomer_states, stability_key)
from writer import FailedMolWriter, mol_with_tags_to_sdf_block

HERE = Path(__file__).resolve().parent


def formal_charge(smiles: str) -> int:
    return Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))


class TestStandardizeAndProtonate(unittest.TestCase):
    def test_standardize_success(self):
        mol, cansmi = standardize_smiles("CCO")
        self.assertIsNotNone(mol)
        self.assertEqual(cansmi, "CCO")

    def test_standardize_invalid_and_empty(self):
        self.assertIsNone(standardize_smiles("X")[0])
        self.assertIsNone(standardize_smiles("")[0])

    def test_standardize_removes_salts(self):
        _, cansmi = standardize_smiles("CCO.[Na+].[Cl-]")
        self.assertEqual(cansmi, "CCO")

    def test_protonation_is_ph_dependent(self):
        acid = Chem.MolFromSmiles("CC(=O)O")
        self.assertEqual(Chem.GetFormalCharge(protonate(acid, pH=7.4)), -1)
        self.assertEqual(Chem.GetFormalCharge(protonate(acid, pH=2.0)), 0)

    def test_protonation_returns_topology_only(self):
        prot = protonate(Chem.MolFromSmiles("CCN"), pH=7.4)
        self.assertEqual(prot.GetNumConformers(), 0)
        self.assertEqual(prot.GetNumAtoms(), 3)       # no explicit hydrogens


class TestDimorphiteVersionWarning(unittest.TestCase):
    def warn_for(self, installed):
        with mock.patch.object(protonate_module, "version", lambda name: installed):
            return dimorphite_version_warning()

    def test_old_versions_trigger_a_warning_that_names_the_fix(self):
        for old in ("2.0.2", "2.0.0", "1.2.5"):
            msg = self.warn_for(old)
            self.assertIn(old, msg)
            self.assertIn("pip install -U", msg)

    def test_recommended_and_newer_versions_do_not_warn(self):
        for ok in ("2.1.0", "2.1.3", "2.10.0", "3.0.0"):
            self.assertIsNone(self.warn_for(ok), ok)

    def test_missing_package_metadata_does_not_crash(self):
        def missing(name):
            raise protonate_module.PackageNotFoundError(name)
        with mock.patch.object(protonate_module, "version", missing):
            self.assertIsNone(dimorphite_version_warning())


class TestForceFields(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(self.mol, randomSeed=42)

    def test_all_three_force_fields(self):
        for name in ("UFF", "MMFF94", "MMFF94s"):
            opt, energy, used = optimize_and_energy(self.mol, name)
            self.assertIsNotNone(opt, name)
            self.assertEqual(used, name)
            self.assertTrue(math.isfinite(energy))

    def test_names_are_case_insensitive(self):
        self.assertEqual(normalize_forcefield_name("mmff94S"), "MMFF94s")
        with self.assertRaises(ValueError):
            normalize_forcefield_name("amber")

    def test_mmff_falls_back_to_uff_and_reports_it(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[Se]C"))    # selenium has no MMFF parameters
        AllChem.EmbedMolecule(mol, randomSeed=1)
        _, used = optimize_conformers(mol, "MMFF94")
        self.assertEqual(used, "UFF")

    def test_requires_explicit_hydrogens(self):
        # the old pipeline silently minimised hydrogen-less structures
        opt, energy, used = optimize_and_energy(Chem.MolFromSmiles("CCO"), "MMFF94")
        self.assertIsNone(opt)
        self.assertIsNone(energy)


class TestTautomers(unittest.TestCase):
    def test_canonical_first_and_flagged(self):
        taut = enumerate_tautomers(Chem.MolFromSmiles("CC(=O)O"))
        self.assertGreaterEqual(len(taut), 1)
        self.assertTrue(taut[0][1])
        self.assertEqual(sum(flag for _, flag in taut), 1)

    def test_stereochemistry_is_preserved(self):
        taut = enumerate_tautomers(Chem.MolFromSmiles("C[C@H](O)C(=O)C"))
        keto = [Chem.MolToSmiles(m) for m, _ in taut if "C(C)=O" in Chem.MolToSmiles(m)
                or "CC(=O)" in Chem.MolToSmiles(m)]
        self.assertTrue(keto)
        self.assertTrue(all("@" in s for s in keto), keto)

    def test_ph_adjustment_drops_tautomers_that_would_change_charge(self):
        # The ionizer is mocked, so this does not depend on the Dimorphite-DL version.
        exotic = Chem.MolFromSmiles("C=C1NC=CN1")
        canonical = Chem.MolFromSmiles("Cc1ncc[nH]1")
        charged = Chem.MolFromSmiles("C=C1[NH2+]C=C[NH2+]1")

        def ionizer(mol, *args, **kwargs):          # wants to charge only the exotic tautomer
            return charged if Chem.MolToSmiles(mol) == Chem.MolToSmiles(exotic) else mol

        with mock.patch.object(tautomer_module, "protonate", ionizer):
            cands = adjust_tautomers_to_ph([(canonical, True), (exotic, False)], pH=7.4)
        self.assertEqual([c.smiles for c in cands], [Chem.MolToSmiles(canonical)])

    def test_canonical_tautomer_is_kept_as_enumerated_if_ionizer_would_change_it(self):
        canonical = Chem.MolFromSmiles("Cc1ncc[nH]1")
        anion = Chem.MolFromSmiles("Cc1ncc[n-]1")
        with mock.patch.object(tautomer_module, "protonate", lambda mol, *a, **k: anion):
            cands = adjust_tautomers_to_ph([(canonical, True)], pH=7.4)
        self.assertEqual([c.smiles for c in cands], [Chem.MolToSmiles(canonical)])
        self.assertTrue(cands[0].is_canonical)

    def test_low_energy_filter_prefers_amide_over_imidic_acid_and_enol(self):
        cands = adjust_tautomers_to_ph(enumerate_tautomers(Chem.MolFromSmiles("CC(=O)Nc1ccccc1")))
        self.assertGreater(len(cands), 1)
        kept = filter_low_energy(cands)
        self.assertEqual([c.smiles for c in kept], ["CC(=O)Nc1ccccc1"])

    def test_low_energy_filter_prefers_keto_over_enol(self):
        cands = adjust_tautomers_to_ph(enumerate_tautomers(Chem.MolFromSmiles("O=C1CCCCC1")))
        self.assertEqual([c.smiles for c in filter_low_energy(cands)], ["O=C1CCCCC1"])

    @staticmethod
    def candidates(*pairs):
        return [TautomerCandidate(Chem.MolFromSmiles(s), s, canon, rank, stability_key(Chem.MolFromSmiles(s)))
                for rank, (s, canon) in enumerate(pairs)]

    def test_equivalent_ring_nh_tautomers_are_both_kept(self):
        cands = self.candidates(("Cc1c[nH]cn1", True), ("Cc1cnc[nH]1", False))
        self.assertEqual(len(filter_low_energy(cands)), 2)

    def kept_smiles(self, smiles, pH=2.0, **kw):
        cands = adjust_tautomers_to_ph(enumerate_tautomers(Chem.MolFromSmiles(smiles)), pH)
        return [c.smiles for c in filter_low_energy(cands, **kw)]

    def test_imide_and_lactam_scaffolds_are_not_turned_into_aromatic_enols(self):
        # RDKit's canonical tautomer of these is an aromatic hydroxy-azole/-furan
        for smiles in ("O=C1CCC(=O)N1",      # succinimide
                       "O=C1CNC(=O)N1",      # hydantoin
                       "O=C1CSC(=O)N1",      # thiazolidine-2,4-dione
                       "O=C1CCC(=O)O1"):     # succinic anhydride
            self.assertEqual(self.kept_smiles(smiles), [smiles], smiles)

    def test_phenol_keeps_its_aromatic_ring(self):
        self.assertEqual(self.kept_smiles("Oc1ccccc1"), ["Oc1ccccc1"])

    def test_rdkit_canonical_tautomer_can_be_forced_and_is_flagged(self):
        smiles = "O=C1CCC(=O)N1"
        cands = adjust_tautomers_to_ph(enumerate_tautomers(Chem.MolFromSmiles(smiles)), 2.0)
        default = filter_low_energy(cands)
        forced = filter_low_energy(cands, always_keep_canonical=True)
        self.assertFalse(any(c.is_canonical for c in default))
        self.assertEqual(sum(c.is_canonical for c in forced), 1)
        self.assertEqual(forced[0].smiles, smiles)          # still ordered most stable first

    def test_canonical_is_first_among_equally_stable_tautomers(self):
        cands = self.candidates(("Cc1cnc[nH]1", False), ("Cc1c[nH]cn1", True))
        kept = filter_low_energy(cands)
        self.assertEqual(len(kept), 2)
        self.assertTrue(kept[0].is_canonical)

    def test_custom_smarts_rule_file_is_used(self):
        # A rule file holding only the keto->enol direction: enol tautomers appear,
        # but nothing else (e.g. no amide->imidic acid) can be generated.
        with tempfile.TemporaryDirectory() as d:
            rules = Path(d) / "rules.txt"
            rules.write_text("1,3 keto/enol f\t[CX4!H0]-[C]=[O]\n")
            smiles = {Chem.MolToSmiles(m) for m, _ in
                      enumerate_tautomers(Chem.MolFromSmiles("CC(=O)C"), transforms_file=str(rules))}
        self.assertEqual(smiles, {"CC(C)=O", "C=C(C)O"})

    def test_full_stage_gives_explicit_hydrogens_and_finite_energies(self):
        # regression test: the old stage-2 output had no hydrogens at all
        scored = prepare_tautomer_states(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"), "MMFF94", 7.4)
        self.assertTrue(scored)
        for taut in scored:
            for state in taut.states:
                self.assertEqual(Chem.AddHs(state.mol).GetNumAtoms(), state.mol.GetNumAtoms())
                self.assertTrue(any(a.GetAtomicNum() == 1 for a in state.mol.GetAtoms()))
                self.assertEqual(state.mol.GetNumConformers(), 1)
                self.assertTrue(math.isfinite(state.energy))


class TestRingConformers(unittest.TestCase):
    def assert_all_chairs(self, state):
        conf = state.mol.GetConformer()
        for ring in saturated_six_rings(state.mol):
            self.assertTrue(is_chair(conf, ring))

    def test_molecule_without_ring_gets_one_state(self):
        states = generate_states(Chem.MolFromSmiles("CCO"))
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].ring_conformer, "NA")

    def test_cyclohexane_is_a_chair_and_one_state(self):
        states = generate_states(Chem.MolFromSmiles("C1CCCCC1"))
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].ring_conformer, "chair")
        self.assert_all_chairs(states[0])

    def test_heterocycles_are_chairs(self):
        for smi in ("C1CCNCC1", "C1COCCN1", "OC1CCNCC1"):
            states = generate_states(Chem.MolFromSmiles(smi))
            self.assertEqual(states[0].ring_conformer, "chair", smi)
            self.assert_all_chairs(states[0])

    def test_symmetric_chairs_are_not_reported_twice(self):
        # both chairs of methylcyclohexane differ by ~1.4 kcal/mol (equatorial wins),
        # and the two enantiomorphic equatorial chairs are the same structure
        states = generate_states(Chem.MolFromSmiles("CC1CCCCC1"))
        self.assertEqual(len(states), 1)

    def test_both_chairs_kept_when_equally_stable(self):
        # cis-1,2-dimethylcyclohexane: the two chairs are mirror-image conformers
        states = generate_states(Chem.MolFromSmiles("C[C@H]1CCCC[C@H]1C"))
        self.assertEqual(len(states), 2)
        self.assertLess(abs(states[0].energy - states[1].energy), 0.5)
        for s in states:
            self.assert_all_chairs(s)

    def test_diequatorial_chair_chosen_for_trans_isomer(self):
        # the diaxial chair is ~1.8 kcal/mol higher and must not be returned
        states = generate_states(Chem.MolFromSmiles("C[C@H]1CCCC[C@@H]1C"))
        self.assertEqual(len(states), 1)

    def test_decalins_cis_has_two_mirror_chairs_trans_is_rigid(self):
        # (R,R) is the more stable trans isomer, (S,S)/(R,S)-type label here is cis
        cis = generate_states(Chem.MolFromSmiles("C1CC[C@H]2CCCC[C@H]2C1"))
        trans = generate_states(Chem.MolFromSmiles("C1CC[C@H]2CCCC[C@@H]2C1"))
        self.assertEqual(len(cis), 2)
        self.assertEqual(len(trans), 1)
        self.assertLess(trans[0].energy, cis[0].energy)

    def test_mirror_states_are_found_regardless_of_sampling_luck(self):
        # with very few random conformers the mirror partner must still appear
        import conformers
        with mock.patch.object(conformers, "_ensemble_size", lambda n: 3):
            for seed in range(1, 6):
                states = generate_states(Chem.MolFromSmiles("C[C@H]1CCCC[C@H]1C"), seed=seed)
                self.assertEqual(len(states), 2, seed)

    def test_chiral_molecule_has_no_mirror_partner(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H]1CCCC[C@@H]1C"))      # chiral trans isomer
        AllChem.EmbedMolecule(mol, randomSeed=3)
        self.assertIsNone(_mirror_partner(mol, 0))

    def test_meso_molecule_has_a_mirror_partner_with_same_stereochemistry(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H]1CCCC[C@H]1C"))       # meso cis isomer
        AllChem.EmbedMolecule(mol, randomSeed=3)
        partner = _mirror_partner(mol, 0)
        self.assertIsNotNone(partner)
        mol.AddConformer(partner, assignId=True)
        smiles = set()
        for cid in (0, 1):
            tmp = Chem.Mol(mol)
            tmp.RemoveAllConformers()
            tmp.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True)
            Chem.AssignStereochemistryFrom3D(tmp)
            smiles.add(Chem.MolToSmiles(Chem.RemoveHs(tmp)))
        self.assertEqual(len(smiles), 1)

    def test_enumeration_can_be_switched_off(self):
        states = generate_states(Chem.MolFromSmiles("C[C@H]1CCCC[C@H]1C"), enumerate_ring_states=False)
        self.assertEqual(len(states), 1)

    def test_ring_forced_out_of_chair_does_not_fail(self):
        states = generate_states(Chem.MolFromSmiles("C1CC2CCC1CC2"))   # bicyclo[2.2.2]octane
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].ring_conformer, "non-chair")

    def test_stereochemistry_survives_3d_generation(self):
        states = generate_states(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
        m = Chem.Mol(states[0].mol)
        Chem.AssignStereochemistryFrom3D(m)
        self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(m)),
                         Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](N)C(=O)O")))

    def test_embedding_failure_raises_conformer_error(self):
        with mock.patch("conformers._embed", return_value=[]):
            with self.assertRaises(ConformerError):
                generate_states(Chem.MolFromSmiles("CCO"))


class TestCharges(unittest.TestCase):
    def setUp(self):
        self.neutral = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.anion = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))

    def test_gasteiger_sums_to_formal_charge(self):
        for mol, net in ((self.neutral, 0), (self.anion, -1)):
            assign_charges(mol, "gasteiger")
            values = [float(x) for x in mol.GetProp(CHARGE_PROPERTY).split()]
            self.assertEqual(len(values), mol.GetNumAtoms())
            self.assertAlmostEqual(sum(values), net, places=6)

    def test_none_writes_nothing(self):
        assign_charges(self.neutral, "none")
        self.assertFalse(self.neutral.HasProp(CHARGE_PROPERTY))

    def test_charges_survive_sdf_round_trip_as_atom_property(self):
        # molscrub/Meeko convention: atom.dprop.PartialCharge -> atom property PartialCharge
        AllChem.EmbedMolecule(self.anion, randomSeed=1)
        assign_charges(self.anion, "gasteiger")
        block = mol_with_tags_to_sdf_block(self.anion, {"Compound_Name": "acetate"})
        back = next(Chem.ForwardSDMolSupplier(io.BytesIO(block.encode()), removeHs=False))
        values = [a.GetDoubleProp("PartialCharge") for a in back.GetAtoms()]
        self.assertEqual(len(values), back.GetNumAtoms())
        self.assertAlmostEqual(sum(values), -1.0, places=6)

    def test_rectify_makes_sum_exact(self):
        q = [0.3333333, 0.3333333, 0.3333334]
        out = rectify_charges(q, 1)
        self.assertAlmostEqual(sum(out), 1.0, places=9)
        self.assertEqual(len(out), 3)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            normalize_charge_model("am1bcc")

    @unittest.skipIf(all(importlib.util.find_spec(m) for m in ("torch", "dgl", "espaloma_charge")),
                     "Espaloma dependencies are installed")
    def test_espaloma_reports_missing_dependencies(self):
        with self.assertRaises(ChargeModelError) as ctx:
            check_charge_model_available("espaloma")
        self.assertIn("pip install", str(ctx.exception))


class TestFailedWriterAndReader(unittest.TestCase):
    def test_no_file_without_failures(self):
        with tempfile.TemporaryDirectory() as d:
            w = FailedMolWriter(Path(d) / "failed.smi")
            w.close()
            self.assertFalse((Path(d) / "failed.smi").exists())

    def test_failed_file_can_be_fed_back_in(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "failed.smi"
            w = FailedMolWriter(path, header_lines=["header"])
            w.write("C1CC", "bad one", "stage1", "standardization failed")
            w.write("[Xx]", "second", "stage2", "no\ttab\nbreaks")
            w.close()
            self.assertEqual(w.count, 2)
            rows = [(s, n) for s, n, _ in read_smiles(str(path))]
        self.assertEqual(rows, [("C1CC", "bad one"), ("[Xx]", "second")])

    def test_three_column_tab_file_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "in.smi"
            path.write_text("CCO\tEthanol\textra\nCCN\tEthylamine\textra\n")
            self.assertEqual([(s, n) for s, n, _ in read_smiles(str(path))],
                             [("CCO", "Ethanol"), ("CCN", "Ethylamine")])


class TestEndToEnd(unittest.TestCase):
    """Runs the real command-line program."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        (tmp / "in.smi").write_text(
            "CCO\tEthanol\n"
            "CC(=O)Nc1ccccc1\tAcetanilide\n"
            "C[C@H]1CCCC[C@H]1C\tcis_dimethylcyclohexane\n"
            "OC1CCNCC1\tHydroxypiperidine\n"
            "X\tBroken\n"
            "CC(=O)[O-]\tAcetate\n"
        )
        cls.out = tmp / "out"
        cls.proc = subprocess.run(
            [sys.executable, str(HERE / "run_ligprep.py"), str(tmp / "in.smi"), "run",
             "-o", str(cls.out), "--num-cpus", "2", "--ph", "7.4",
             "-ff", "MMFF94s", "--charge_model", "gasteiger"],
            capture_output=True, text=True, cwd=str(HERE))
        cls.first = [m for m in Chem.SDMolSupplier(str(cls.out / "run.sdf"), removeHs=False)]
        cls.second = [m for m in Chem.SDMolSupplier(str(cls.out / "run_taut.sdf"), removeHs=False)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_program_succeeds(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)

    def test_output_files_exist(self):
        names = {p.name for p in self.out.iterdir()}
        self.assertTrue({"run.sdf", "run_taut.sdf", "run_failed.smi"} <= names, names)
        self.assertTrue(any(n.startswith("run_") and n.endswith(".log") for n in names))
        self.assertNotIn("invalid_smiles.csv", names)

    def test_failed_molecule_is_reported_in_smi_file(self):
        text = (self.out / "run_failed.smi").read_text()
        self.assertIn("X\tBroken\tstage1", text)
        self.assertEqual(len(self.first), 5)               # 6 inputs - 1 failure

    def test_first_sdf_title_is_compound_name(self):
        self.assertEqual({m.GetProp("_Name") for m in self.first},
                         {"Ethanol", "Acetanilide", "cis_dimethylcyclohexane",
                          "Hydroxypiperidine", "Acetate"})

    def test_second_sdf_has_explicit_hydrogens(self):
        self.assertTrue(self.second)
        for m in self.second:
            self.assertEqual(Chem.AddHs(m).GetNumAtoms(), m.GetNumAtoms(), m.GetProp("_Name"))
            self.assertGreater(sum(a.GetAtomicNum() == 1 for a in m.GetAtoms()), 0)

    def test_second_sdf_title_is_tautomer_name_and_tags_present(self):
        for m in self.second:
            self.assertEqual(m.GetProp("_Name"), m.GetProp("Tautomer_Name"))
            for tag in ("UUID", "Original_SMILES", "pH", "Force_field", "Generation_Date",
                        "Is_Canonical_Tautomer", "Energy", "Optimization_ForceField",
                        "Ring_Conformer", "Charge_Model"):
                self.assertTrue(m.HasProp(tag), tag)
            self.assertEqual(m.GetProp("Optimization_ForceField"), "MMFF94s")
            self.assertEqual(m.GetProp("Charge_Model"), "gasteiger")
            self.assertTrue(math.isfinite(float(m.GetProp("Energy"))))

    def test_charges_are_per_atom_and_sum_to_net_charge(self):
        for m in self.second:
            q = [a.GetDoubleProp("PartialCharge") for a in m.GetAtoms()]
            self.assertEqual(len(q), m.GetNumAtoms())
            self.assertAlmostEqual(sum(q), Chem.GetFormalCharge(m), places=6)

    def test_ring_states_are_enumerated_and_uniquely_named(self):
        ring = [m for m in self.second if m.GetProp("Parent_compound") == "cis_dimethylcyclohexane"]
        self.assertEqual(len(ring), 2)
        self.assertEqual({m.GetProp("_Name") for m in ring},
                         {"cis_dimethylcyclohexane_taut0001_ring1",
                          "cis_dimethylcyclohexane_taut0001_ring2"})
        self.assertTrue(all(m.GetProp("Ring_Conformer") == "chair" for m in ring))

    def test_amide_gives_only_the_amide_tautomer(self):
        amide = [m for m in self.second if m.GetProp("Parent_compound") == "Acetanilide"]
        self.assertEqual(len(amide), 1)
        self.assertEqual(amide[0].GetProp("Is_Canonical_Tautomer"), "1")

    def test_invalid_charge_model_is_rejected(self):
        res = subprocess.run([sys.executable, str(HERE / "run_ligprep.py"), "x.smi", "y",
                              "--charge_model", "am1bcc"], capture_output=True, text=True, cwd=str(HERE))
        self.assertNotEqual(res.returncode, 0)

    def test_espaloma_without_dependencies_fails_fast(self):
        if all(importlib.util.find_spec(m) for m in ("torch", "dgl", "espaloma_charge")):
            self.skipTest("Espaloma dependencies are installed")
        res = subprocess.run([sys.executable, str(HERE / "run_ligprep.py"), str(Path(self.tmp.name) / "in.smi"),
                              "esp", "-o", str(Path(self.tmp.name) / "esp"), "--charge_model", "espaloma"],
                             capture_output=True, text=True, cwd=str(HERE))
        self.assertEqual(res.returncode, 2)
        self.assertFalse((Path(self.tmp.name) / "esp" / "esp.sdf").exists())


class TestExampleFile(unittest.TestCase):
    """The shipped example_input.smi must prepare cleanly, as its header promises."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.proc = subprocess.run(
            [sys.executable, str(HERE / "run_ligprep.py"), str(HERE / "example_input.smi"), "demo",
             "-o", str(cls.out), "--num-cpus", "2", "--charge_model", "gasteiger"],
            capture_output=True, text=True, cwd=str(HERE))
        cls.first = list(Chem.SDMolSupplier(str(cls.out / "demo.sdf"), removeHs=False))
        cls.second = list(Chem.SDMolSupplier(str(cls.out / "demo_taut.sdf"), removeHs=False))
        cls.by = {}
        for m in cls.second:
            cls.by.setdefault(m.GetProp("Parent_compound"), []).append(m)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_molecule_succeeds_and_nothing_is_reported_failed(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        self.assertEqual(len(self.first), 35)
        self.assertFalse((self.out / "demo_failed.smi").exists())
        self.assertGreaterEqual(len(self.second), 35)

    def test_all_records_have_hydrogens_and_per_atom_charges(self):
        for m in self.second:
            self.assertEqual(Chem.AddHs(m).GetNumAtoms(), m.GetNumAtoms())
            self.assertEqual(len([a.GetDoubleProp("PartialCharge") for a in m.GetAtoms()]), m.GetNumAtoms())

    def test_ring_examples_behave_as_documented(self):
        counts = {name: len(ms) for name, ms in self.by.items()}
        self.assertEqual(counts["cis-1,2-Dimethylcyclohexane"], 2)
        self.assertEqual(counts["trans-1,2-Dimethylcyclohexane"], 1)
        self.assertEqual(counts["cis-Decalin"], 2)
        self.assertEqual(counts["trans-Decalin"], 1)
        self.assertEqual(self.by["Bicyclo222octane"][0].GetProp("Ring_Conformer"), "non-chair")
        for name in ("Cyclohexane", "Tetrahydropyran", "L-Menthol", "Morpholine"):
            self.assertEqual(self.by[name][0].GetProp("Ring_Conformer"), "chair", name)

    def test_tautomer_examples_behave_as_documented(self):
        self.assertEqual(self.by["Acetanilide"][0].GetProp("Tautomer_SMILES"), "CC(=O)Nc1ccccc1")
        self.assertEqual(self.by["Acetylacetone"][0].GetProp("Tautomer_SMILES"), "CC(=O)CC(C)=O")
        self.assertEqual(self.by["Nicotinamide"][0].GetProp("Tautomer_SMILES"), "NC(=O)c1cccnc1")

    def test_salts_are_stripped_and_stereo_kept(self):
        self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(self.by["Ethylamine_hydrochloride"][0])).count("."), 0)
        self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(self.by["Sodium_acetate"][0])).count("."), 0)
        self.assertIn("@", self.by["L-Alanine_S"][0].GetProp("Tautomer_SMILES"))

    def test_nameless_entries_are_named_after_their_line(self):
        names = {m.GetProp("Parent_compound") for m in self.second}
        self.assertTrue({"mol_74", "mol_75"} <= names, names)

    @unittest.skipIf(dimorphite_version_warning() is not None, "needs Dimorphite-DL >= 2.1.0")
    def test_protonation_and_multi_tautomer_examples(self):
        self.assertEqual(len(self.second), 41)        # the count promised in the file header
        charge = lambda n: Chem.GetFormalCharge(self.by[n][0])
        self.assertEqual(charge("Acetic_acid"), -1)
        self.assertEqual(charge("Ethylamine"), +1)
        self.assertEqual(charge("Glycine_zwitterion"), 0)
        for name in ("4-Methylimidazole", "3-Methylpyrazole", "Adenine", "Theophylline"):
            flags = sorted(m.GetProp("Is_Canonical_Tautomer") for m in self.by[name])
            self.assertEqual(flags, ["0", "1"], name)


if __name__ == "__main__":
    unittest.main()
