from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.PICIO import read_PIC_seq
from Bio.PDB.ic_rebuild import write_PDB

def build_ideal_alpha_helix(seq, out_pdb="ideal_alpha_helix.pdb",
                            phi=-57.0, psi=-47.0, omega=180.0):
    # 先生成默认（mostly helical）内部坐标结构
    structure = read_PIC_seq(
        SeqRecord(
            Seq(seq),
            id="AHLX",
            description="idealized alpha helix"
        )
    )

    chain = structure[0]["A"]

    # 对每个残基的内部坐标进行调整
    for residue in chain:
        ric = residue.internal_coord
        if ric is None:
            continue

        # 首残基通常没有 phi；末残基通常没有 psi；omega 也可能在链端缺失
        if ric.get_angle("phi") is not None:
            ric.set_angle("phi", phi)
        if ric.get_angle("psi") is not None:
            ric.set_angle("psi", psi)
        if ric.get_angle("omega") is not None:
            # Biopython 里常见默认值是接近 179/180
            ric.set_angle("omega", omega)

    # 从内部坐标重建 3D 坐标
    structure.internal_to_atom_coordinates()

    # 输出 PDB
    write_PDB(structure, out_pdb)
    return structure

# 示例
build_ideal_alpha_helix("AAAAAKAAAAAKAAAAA", "ideal_alpha_helix.pdb")