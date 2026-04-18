"""PDB writer helpers."""

from density2sse.io.pdb_io import _chain_id_for_index


def test_chain_id_62_symbols():
    assert _chain_id_for_index(0) == "A"
    assert _chain_id_for_index(25) == "Z"
    assert _chain_id_for_index(26) == "a"
    assert _chain_id_for_index(51) == "z"
    assert _chain_id_for_index(52) == "0"
    assert _chain_id_for_index(61) == "9"


def test_chain_id_rejects_beyond_62():
    import pytest

    from density2sse.io import pdb_io

    with pytest.raises(ValueError, match="At most 62"):
        pdb_io._chain_id_for_index(62)
