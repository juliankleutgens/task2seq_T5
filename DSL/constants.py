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


F = False
T = True

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
TEN = 10

NEG_ONE = -1
NEG_TWO = -2

DOWN = (1, 0)
RIGHT = (0, 1)
UP = (-1, 0)
LEFT = (0, -1)

ORIGIN = (0, 0)
UNITY = (1, 1)
NEG_UNITY = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)

ZERO_BY_TWO = (0, 2)
TWO_BY_ZERO = (2, 0)
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)