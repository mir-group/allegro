# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
def to_int_list(arg):
    # converts str: "64 64 64" to List[int]: [64, 64, 64]
    # or int: 64 to List[int]: [64]
    # to simplify parsing of list inputs when using 3rd party code
    # e.g. to pass inputs to wandb sweep configs
    if isinstance(arg, str):
        return [int(x) for x in arg.split()]
    elif isinstance(arg, int):
        return [arg]
    else:
        return arg
