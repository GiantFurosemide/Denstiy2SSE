"""Write multi-chain backbone PDB files."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


def _element_for(name: str) -> str:
    n = name.strip().upper()
    if n == "CA":
        return "C"
    if n == "N":
        return "N"
    if n == "C":
        return "C"
    if n == "O":
        return "O"
    return n[0]


def _fullname(name: str) -> str:
    n = name.strip()
    if len(n) == 1:
        return f" {n}  "
    if len(n) == 2:
        return f" {n} "
    if len(n) == 3:
        return f" {n}"
    return n[:4]


def write_backbone_pdb(
    chains: List[List[List[Tuple[str, np.ndarray]]]],
    out_path: str,
    title: str = "density2sse",
) -> None:
    """
    ``chains``: list of chains; each chain is a list of residues; each residue is
    ``[(atom_name, (3,)), ...]``.
    """
    structure = Structure(title[:40])
    model = Model(0)
    structure.add(model)
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    serial = 1
    for ci, chain_residues in enumerate(chains):
        cid = chain_ids[ci % len(chain_ids)]
        chain = Chain(cid)
        model.add(chain)
        for ri, residue_atoms in enumerate(chain_residues, start=1):
            res_id = (" ", ri, " ")
            residue = Residue(res_id, "ALA", " ")
            chain.add(residue)
            for aname, xyz in residue_atoms:
                coord = np.asarray(xyz, dtype=np.float64)
                fn = _fullname(aname)
                atom = Atom(
                    aname.strip(),
                    coord,
                    0.0,
                    1.0,
                    " ",
                    fn,
                    serial,
                    element=_element_for(aname),
                )
                residue.add(atom)
                serial += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def helices_to_pdb_file(
    helix_atom_blocks: Iterable[np.ndarray],
    out_path: str,
    title: str = "density2sse",
) -> None:
    """
    ``helix_atom_blocks``: iterable of arrays shaped ``(N_res, 4, 3)`` in order N, CA, C, O.
    """
    names = ("N", "CA", "C", "O")
    chains: List = []
    for block in helix_atom_blocks:
        b = np.asarray(block, dtype=np.float64)
        residues = []
        for ri in range(b.shape[0]):
            residues.append([(names[a], b[ri, a]) for a in range(4)])
        chains.append(residues)
    write_backbone_pdb(chains, out_path, title=title)
