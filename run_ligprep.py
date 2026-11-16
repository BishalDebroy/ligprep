"""
main.py - Ligand Preparation Pipeline.
Basic usage: python run_ligprep.py example_input.smi output_name --ph 7.4 -o results

Stage 1: SMILES -> standardize -> protonate (Dimorphite-DL) -> 3D (chair-corrected rings)
Stage 2: protonated 3D -> pH-dependent low-energy tautomers (canonical flagged)
         -> 3D with explicit hydrogens and ring-flip states -> partial charges
"""
import argparse
import io
import logging
import multiprocessing
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

from rdkit import Chem

from reader import read_smiles, read_sdf_blocks
from standardize import standardize_smiles
from protonate import protonate, dimorphite_version_warning
from conformers import generate_states, ConformerError
from tautomer import prepare_tautomer_states
from optimizer import SUPPORTED_FORCE_FIELDS, normalize_forcefield_name
from charges import (CHARGE_MODELS, ChargeModelError, assign_charges,
                     check_charge_model_available, normalize_charge_model)
from metadata import first_sdf_tags, second_sdf_tags
from writer import mol_with_tags_to_sdf_block, write_block_to_handle, FailedMolWriter
from utils import setup_logging, unique_id, current_date_tag, memory_usage_mb, log_memory, start_queue_listener, worker_logging_init

__version__ = "1.3.0"

logger = logging.getLogger("ligprep")

# A failure is reported as (smiles, name, reason).
Failure = Tuple[str, str, str]


# ---------------------------------------------------------------------
# Worker functions for parallel stages
# ---------------------------------------------------------------------
def process_smiles_worker(
        args: Tuple[str, str, int, float, Optional[float], Optional[float], int, str]
) -> Tuple[Optional[str], Optional[Failure]]:
    """
    Stage 1 for a single SMILES entry.
    Returns (sdf_block, None) on success or (None, (smiles, name, reason)) on failure.
    """
    smi, name, line_no, pH, min_pH, max_pH, max_states, force_field = args
    display_name = name if name else f"mol_{line_no}"

    def fail(reason: str) -> Tuple[None, Failure]:
        logger.warning(f"Failed molecule at line {line_no}: {smi} - {reason}")
        return None, (smi, display_name, reason)

    try:
        mol, cansmi = standardize_smiles(smi)
        if mol is None:
            return fail("standardization failed (invalid or empty SMILES)")

        # Protonation (top-ranked state only)
        prot_mol = protonate(mol, pH, min_pH, max_pH, max_states)
        if prot_mol is None:
            return fail("protonation failed")

        # 3D structure: lowest-energy chair arrangement, one record per molecule
        try:
            state = generate_states(prot_mol, force_field, enumerate_ring_states=False)[0]
        except ConformerError as e:
            return fail(f"3D generation failed: {e}")

        # Generate metadata
        tags = first_sdf_tags(
            compound_name=display_name,
            original_smi=smi,
            canonical_smi=cansmi,
            force_field=state.force_field,
            ph=pH,
            uuid=unique_id(),
            generation_date=current_date_tag()
        )
        return mol_with_tags_to_sdf_block(state.mol, tags), None
    except Exception as e:
        return fail(f"unexpected error: {e}")


def process_tautomer_worker(
        args: Tuple[str, int, str, float, Optional[float], Optional[float], int, str]
) -> Tuple[List[str], Optional[Failure]]:
    """
    Stage 2 for one stage-1 SDF block: pH-dependent low-energy tautomers, each with
    explicit hydrogens, ring states and optional partial charges.
    Returns (list of SDF blocks, None) or ([], (smiles, name, reason)).
    """
    block_str, entry_idx, force_field, pH, min_pH, max_pH, max_states, charge_model = args
    try:
        supplier = Chem.ForwardSDMolSupplier(io.BytesIO(block_str.encode("utf-8")), removeHs=False)
        mol = next(supplier, None)
    except Exception as e:
        logger.error(f"Could not parse SDF entry {entry_idx}: {e}")
        return [], (f"[SDF entry {entry_idx}]", f"entry_{entry_idx}", f"could not parse SDF entry: {e}")
    if mol is None:
        logger.error(f"Could not parse SDF entry {entry_idx}")
        return [], (f"[SDF entry {entry_idx}]", f"entry_{entry_idx}", "could not parse SDF entry")

    # Recover parent tags from the SDF properties (kept as strings)
    parent_tags = {k: mol.GetProp(k) for k in mol.GetPropNames()}
    parent_compound = parent_tags.get("Compound_Name", f"mol_{entry_idx}")
    parent_smiles = parent_tags.get("Original_SMILES", "")

    def fail(reason: str) -> Tuple[List[str], Failure]:
        logger.warning(f"Failed molecule {parent_compound}: {reason}")
        return [], (parent_smiles, parent_compound, reason)

    try:
        scored = prepare_tautomer_states(Chem.RemoveHs(mol), force_field, pH, min_pH, max_pH, max_states)
    except Exception as e:
        return fail(f"tautomer stage failed: {e}")
    if not scored:
        return fail("no tautomer could be converted to a 3D structure")

    blocks = []
    for taut_num, taut in enumerate(scored, start=1):
        for ring_num, state in enumerate(taut.states, start=1):
            try:
                assign_charges(state.mol, charge_model)
            except ChargeModelError as e:
                logger.warning(f"Skipping a state of {parent_compound}_taut{taut_num:04d}: {e}")
                continue
            taut_name = f"{parent_compound}_taut{taut_num:04d}"
            if len(taut.states) > 1:
                taut_name += f"_ring{ring_num}"
            tags = second_sdf_tags(
                parent_tags=parent_tags,
                tautomer_name=taut_name,
                tautomer_number=taut_num,
                is_canonical=taut.candidate.is_canonical,
                energy=state.energy,
                energy_unit="kcal/mol",
                opt_force_field=state.force_field,
                tautomer_smiles=taut.candidate.smiles,
                relative_energy=state.energy - taut.states[0].energy,
                ring_conformer=state.ring_conformer,
                ring_state_number=ring_num,
                charge_model=charge_model,
            )
            blocks.append(mol_with_tags_to_sdf_block(state.mol, tags))
    if not blocks:
        return fail(f"partial-charge assignment ({charge_model}) failed for every state")
    return blocks, None


# ---------------------------------------------------------------------
# Command-line helpers
# ---------------------------------------------------------------------
def _forcefield_arg(value: str) -> str:
    try:
        return normalize_forcefield_name(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _charge_model_arg(value: str) -> str:
    try:
        return normalize_charge_model(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ligand preparation pipeline")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input_smi", help="input SMILES file")
    parser.add_argument("output_basename", help="Basename for output SDF files")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for protonation (default 7.4)")
    parser.add_argument("--min_ph", type=float, help="Minimum pH for protonation range")
    parser.add_argument("--max_ph", type=float, help="Maximum pH for protonation range")
    parser.add_argument("-ff", default="MMFF94", type=_forcefield_arg, choices=list(SUPPORTED_FORCE_FIELDS),
                        help="Force field: UFF, MMFF94 or MMFF94s (default MMFF94; falls back to UFF "
                             "for molecules MMFF cannot parametrise)")
    parser.add_argument("--charge_model", default="none", type=_charge_model_arg, choices=list(CHARGE_MODELS),
                        help="Partial charges written to the tautomer SDF: gasteiger, espaloma or none "
                             "(default none)")
    parser.add_argument("--num-cpus", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use")
    parser.add_argument("-o", default=".", help="Directory for output files")
    parser.add_argument("--overwrite", action="store_true", help="Overwriting existing output files")
    parser.add_argument("-l", dest="log_file", default=None, help="Log file path (default: <output_dir>/<output_basename>_<DDMONYY>.log)")
    parser.add_argument("-v", action="store_true", help="Verbose logging")
    parser.add_argument("--max_protonation_states", type=int, default=1, help="Number of protonation states to keep (default 1)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for parallel processing")
    args = parser.parse_args()

    # Output directory must exist before we can default the log path into it
    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if args.log_file:
        log_path = args.log_file
    else:
        log_path = str(out_dir / f"{args.output_basename}_{current_date_tag()}.log")
    logger = setup_logging(log_path, args.v)
    logger.info(f"Ligand Preparation Pipeline v{__version__}")
    logger.info("Ligand Preparation Started")
    logger.info(f"Settings: pH={args.ph} force_field={args.ff} charge_model={args.charge_model}")
    start_time = datetime.now()
    log_memory(logger, "Start")
    dimorphite_warning = dimorphite_version_warning()
    if dimorphite_warning:
        logger.warning(dimorphite_warning)

    # Fail fast if the requested charge model cannot be used
    try:
        check_charge_model_available(args.charge_model)
    except ChargeModelError as e:
        logger.error(str(e))
        sys.exit(2)

    # Multiprocessing-safe logging: workers route log records through this
    # queue instead of writing directly to the file/console handles they'd
    # otherwise inherit via fork (which causes concurrent-write/fd errors).
    log_manager = multiprocessing.Manager()
    log_queue = log_manager.Queue(-1)
    log_listener = start_queue_listener(logger, log_queue)

    first_sdf_path = out_dir / f"{args.output_basename}.sdf"
    second_sdf_path = out_dir / f"{args.output_basename}_taut.sdf"
    failed_path = out_dir / f"{args.output_basename}_failed.smi"

    # Overwrite check
    if not args.overwrite:
        for p in [first_sdf_path, second_sdf_path]:
            if p.exists():
                logger.error(f"Output file {p} already exists. Use --overwrite to replace.")
                sys.exit(1)

    failed_writer = FailedMolWriter(failed_path, header_lines=[
        f"ligprep v{__version__}: molecules that could not be prepared",
        "columns: SMILES<TAB>name<TAB>stage: reason  (can be re-submitted as a .smi input)",
    ])

    # ------- Stage 1: SMILE → standardize + protonated SDF ---------
    logger.info("Stage 1: SMILE to 3D generation")

    pool = multiprocessing.Pool(
        processes=args.num_cpus,
        initializer=worker_logging_init,
        initargs=(log_queue, args.v),
    )

    # Task generator streams the input file so memory stays low
    def task_generator():
        for smi, name, lineno in read_smiles(args.input_smi):
            yield (smi, name, lineno, args.ph, args.min_ph, args.max_ph, args.max_protonation_states, args.ff)

    total_processed = 0
    total_skipped = 0

    # One file handle for the whole stage (no reopening per molecule)
    with open(first_sdf_path, "w", encoding="utf-8") as first_sdf_fh:
        for block, failure in pool.imap_unordered(process_smiles_worker, task_generator(), chunksize=args.batch_size):
            if failure is not None:
                failed_writer.write(failure[0], failure[1], "stage1", failure[2])
                total_skipped += 1
                continue
            if block:
                write_block_to_handle(first_sdf_fh, block)
                total_processed += 1

    pool.close()
    pool.join()

    logger.info(f"Stage 1 finished. processed {total_processed} molecules, skipped {total_skipped} molecules.")
    log_memory(logger, "after stage 1")

    # --------------- Stage 2: Tautomer generation ------------------
    logger.info("Stage 2: Tautomer enumeration and optimizaton")

    if not first_sdf_path.exists() or first_sdf_path.stat().st_size == 0:
        logger.warning("No multi-ligand 3D SDF found, skipping tautomer generation.")
    else:
        pool2 = multiprocessing.Pool(
            processes=args.num_cpus,
            initializer=worker_logging_init,
            initargs=(log_queue, args.v),
        )

        def sdf_task_gen():
            for blk, idx in read_sdf_blocks(str(first_sdf_path)):
                yield (blk, idx, args.ff, args.ph, args.min_ph, args.max_ph,
                       args.max_protonation_states, args.charge_model)

        total_tautomers = 0
        stage2_failed = 0
        with open(second_sdf_path, "w", encoding="utf-8") as fh_out:
            for taut_blocks, failure in pool2.imap_unordered(process_tautomer_worker, sdf_task_gen(), chunksize=args.batch_size):
                if failure is not None:
                    failed_writer.write(failure[0], failure[1], "stage2", failure[2])
                    stage2_failed += 1
                    continue
                for blk in taut_blocks:
                    write_block_to_handle(fh_out, blk)
                    total_tautomers += 1

        pool2.close()
        pool2.join()
        logger.info(f"Stage 2 finished. Total tautomers written: {total_tautomers} "
                    f"(molecules failed in stage 2: {stage2_failed})")

    # --------------- Failed molecules ------------------
    failed_writer.close()
    if failed_writer.count:
        logger.info(f"{failed_writer.count} molecule(s) failed; see {failed_path}")
    else:
        logger.info("No molecules failed.")
        if failed_path.exists():
            failed_path.unlink()   # stale file from an earlier run

    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"Ligand Preparation completed. elapsed time: {elapsed}")
    log_memory(logger, "final")
    logger.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_listener.stop()


if __name__ == "__main__":
    main()
