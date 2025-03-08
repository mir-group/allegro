import torch
from nequip.nn import with_edge_vectors_
from nequip.data import AtomicDataDict
from allegro._compile import allegro_data_settings


def test_pair_allegro_ghost_consistency(Cu_bulk):
    _, data = Cu_bulk
    pair_allegro_data = allegro_data_settings(data.copy())
    pair_allegro_data = with_edge_vectors_(pair_allegro_data)
    data = with_edge_vectors_(data)
    neighbors_outside_cell = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].abs().sum(-1) != 0
    pa_len = pair_allegro_data[AtomicDataDict.EDGE_LENGTH_KEY]
    cell_len = data[AtomicDataDict.EDGE_LENGTH_KEY]
    cell_len = torch.cat(
        [cell_len[~neighbors_outside_cell], cell_len[neighbors_outside_cell]], 0
    )
    assert torch.allclose(pa_len, cell_len)
