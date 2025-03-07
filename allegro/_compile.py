from nequip.data import AtomicDataDict
from nequip.scripts._compile_utils import (
    LMP_OUTPUTS,
    single_frame_batch_map_settings,
    single_frame_data_settings,
    register_compile_targets,
)

PAIR_ALLEGRO_INPUTS = [
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
]


PAIR_ALLEGRO_TARGET = {
    "input": PAIR_ALLEGRO_INPUTS,
    "output": LMP_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": single_frame_data_settings,
}

register_compile_targets({"pair_allegro": PAIR_ALLEGRO_TARGET})
