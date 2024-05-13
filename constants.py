from typing import Dict, List, NamedTuple

import numpy as np
from matplotlib import colors
from numpy.typing import NDArray

ALLOWED_COLORS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

COLORMAP = colors.ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
        "#FFFFFF",
    ]
)
NORM = colors.Normalize(vmin=0, vmax=10)


class DoesNotFitError(Exception):
    pass


# Example: {"train": [{"input": np.array, "output": np.array}]}
TASK_DICT = Dict[str, List[Dict[str, NDArray[np.uint8]]]]

Example = NamedTuple(
    "Example", [("input", NDArray[np.uint8]), ("output", NDArray[np.uint8])]
)
Task = List[Example]
