# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict
from nequip.scripts._compile_utils import (
    LMP_OUTPUTS,
    single_frame_batch_map_settings,
    register_compile_targets,
)

PAIR_ALLEGRO_INPUTS = [
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
]


def allegro_data_settings(data):
    assert AtomicDataDict.num_frames(data) == 1

    # because of the 0/1 specialization problem,
    # and the fact that the LAMMPS pair style (and ASE) requires `num_frames=1`
    # we need to augment to data to remove the `BATCH_KEY` and `NUM_NODES_KEY`
    # to take more optimized code paths
    if AtomicDataDict.BATCH_KEY in data:
        data.pop(AtomicDataDict.BATCH_KEY)
        data.pop(AtomicDataDict.NUM_NODES_KEY)

    if AtomicDataDict.EDGE_CELL_SHIFT_KEY in data:
        # if data has PBC, we convert it to the pair_allegro format with ghost atoms and not edge cell shifts
        # == get convenience variables ==
        pos = data[AtomicDataDict.POSITIONS_KEY]
        edge_idx = data[AtomicDataDict.EDGE_INDEX_KEY]
        cell_shift = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        cell = data[AtomicDataDict.CELL_KEY]

        # mask for neighbors that lie outside of 000 cell
        neighbors_outside_cell = cell_shift.abs().sum(-1) != 0

        # get ghost atom positions and atom types
        edge_idx_outside_cell = edge_idx[:, neighbors_outside_cell]
        pos_outside_cell = torch.index_select(
            pos, 0, edge_idx_outside_cell[1]
        ) + torch.mm(cell_shift[neighbors_outside_cell], cell.view(3, 3))
        type_outside_cell = torch.index_select(
            data[AtomicDataDict.ATOM_TYPE_KEY], 0, edge_idx_outside_cell[1]
        )

        # set ghost atom neighbor indices
        edge_idx_outside_cell[1] = torch.arange(
            pos.size(0), pos.size(0) + pos_outside_cell.size(0), device=pos.device
        )
        edge_idx_inside_cell = edge_idx[:, ~neighbors_outside_cell]

        # update AtomicDataDict
        data[AtomicDataDict.POSITIONS_KEY] = torch.cat([pos, pos_outside_cell], dim=0)
        data[AtomicDataDict.ATOM_TYPE_KEY] = torch.cat(
            [data[AtomicDataDict.ATOM_TYPE_KEY], type_outside_cell], dim=0
        )
        data[AtomicDataDict.EDGE_INDEX_KEY] = torch.cat(
            [edge_idx_inside_cell, edge_idx_outside_cell], dim=1
        )
        data.pop(AtomicDataDict.EDGE_CELL_SHIFT_KEY)
        data.pop(AtomicDataDict.CELL_KEY)

    return data


PAIR_ALLEGRO_TARGET = {
    "input": PAIR_ALLEGRO_INPUTS,
    "output": LMP_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": allegro_data_settings,
}
register_compile_targets({"pair_allegro": PAIR_ALLEGRO_TARGET})
