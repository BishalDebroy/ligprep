"""
writer.py - Write mol blocks to an SDF, and failed molecules to a .smi file.
"""
import io
from typing import Dict, Iterable, IO, List, Optional
from pathlib import Path
from rdkit import Chem

def mol_with_tags_to_sdf_block(mol: Chem.Mol, tags: Dict[str, str]) -> str:
    """Convert an RDKit Mol and a dict of tags into a single SDF block string.

    The SDF title line (line 1 of each block) is set to the Tautomer_Name
    if present (second SDF), otherwise to the Compound_Name (first SDF).
    """
    title = tags.get("Tautomer_Name") or tags.get("Compound_Name") or ""
    mol.SetProp("_Name", str(title))

    # set all properties
    for key, value in tags.items():
        mol.SetProp(key, str(value))
    sio = io.StringIO()
    w = Chem.SDWriter(sio)
    w.write(mol)
    w.flush()
    w.close()
    block = sio.getvalue()
    return block

def format_sdf_block(block: str) -> str:
    """Ensure a single SDF block string is properly terminated with '$$$$'."""
    if not block.strip().endswith("$$$$"):
        block = block.rstrip() + "\n$$$$\n"
    return block

def write_block_to_handle(fh: IO[str], block: str) -> None:
    """
    Write a single, already-generated SDF block to an already-open file
    handle. Use this in hot loops (e.g. once per molecule across a large
    library) instead of write_sdf_blocks, which opens and closes the file
    on every call - reopening a file hundreds of thousands of times is
    slow, especially on non-native filesystems (e.g. WSL/DrvFs mounts).
    """
    fh.write(format_sdf_block(block))

def write_sdf_blocks(blocks: Iterable[str], output_path: str, append: bool = False) -> None:
    """
    Write an iterable of SDF block strings to a file in a single open/close.
    Convenience wrapper for writing a full batch at once; for writing many
    blocks one at a time as they're produced, keep the file open yourself
    and call write_block_to_handle() per block instead.
    """
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as fh:
        for blk in blocks:
            write_block_to_handle(fh, blk)


def _clean(field: str) -> str:
    """Make a value safe for a tab-separated line."""
    return " ".join(str(field).split())


class FailedMolWriter:
    """Collects molecules that could not be prepared into a .smi file.

    Each line is ``SMILES<TAB>name<TAB>stage: reason``. The file is created
    lazily on the first failure, so a clean run leaves no file behind. It can be
    fed straight back into the pipeline (the reason column is ignored when reading).
    """

    def __init__(self, path, header_lines: Optional[List[str]] = None):
        self.path = Path(path)
        self.header_lines = header_lines or []
        self.count = 0
        self._fh: Optional[IO[str]] = None

    def write(self, smiles: str, name: str, stage: str, reason: str) -> None:
        if self._fh is None:
            self._fh = open(self.path, "w", encoding="utf-8")
            for line in self.header_lines:
                self._fh.write(f"# {line}\n")
        self._fh.write(f"{_clean(smiles) or '[EMPTY]'}\t{_clean(name) or '-'}\t{_clean(stage)}: {_clean(reason)}\n")
        self.count += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
