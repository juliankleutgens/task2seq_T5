from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable
)

Boolean = bool
Integer = int
IntegerTuple = Tuple[Integer, Integer]
Numerical = Union[Integer, IntegerTuple]
IntegerSet = FrozenSet[Integer]
Grid = Tuple[Tuple[Integer]]
Cell = Tuple[Integer, IntegerTuple]
Object = FrozenSet[Cell]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerTuple]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]



types_dict = {
    'Container': Container,
    'Callable': Callable,
    'Any': Any,
    'FrozenSet': FrozenSet,
    'Tuple': Tuple,
    'Grid': Grid,
    'IntegerTuple': IntegerTuple,
    'Numerical': Numerical,
    'IntegerSet': IntegerSet,
    'Cell': Cell,
    'Object': Object,
    'Objects': Objects,
    'Indices': Indices,
    'IndicesSet': IndicesSet,
    'Patch': Patch,
    'Element': Element,
    'Piece': Piece,
    'Boolean': Boolean,
    'Integer': Integer,
    'TupleTuple': TupleTuple,
    'ContainerContainer': ContainerContainer,
}


arc_types_mapper = {
    Container: {Container, ContainerContainer, Tuple, TupleTuple, Grid, FrozenSet, Object, Objects, Indices, IndicesSet, Cell},
    Callable: {Callable},
    Any: set(types_dict.values()),
    FrozenSet: {FrozenSet, Object, Objects, Indices, IndicesSet, IntegerSet},
    Tuple: {Tuple, TupleTuple, Cell},
    Grid: {Grid},
    IntegerTuple: {IntegerTuple},
    Numerical: {Numerical, Integer, IntegerTuple},
    IntegerSet: {IntegerSet},
    Cell: {Cell},
    Object: {Object},
    Objects: {Objects},
    Indices: {Indices},
    IndicesSet: {IndicesSet},
    Patch: {Patch, Object, Indices},
    Integer: {Integer},
    Boolean: {Boolean},
    Element: {Element, Object, Grid},
    Piece: {Piece, Grid, Patch, Indices, Object},
    TupleTuple: {TupleTuple},
    ContainerContainer: {ContainerContainer, TupleTuple, Objects, IndicesSet}
}


base_types = {
    Boolean, Integer, IntegerTuple, IntegerSet,
    Callable, Grid, Cell, Object, Objects, Indices, IndicesSet
}

non_base_type = {
    Numerical, List, Union, Tuple, Any, Container, FrozenSet, Iterable,
    Patch, Element, Piece, TupleTuple, ContainerContainer
}


child_mapper = {
    Objects: Object,
    IntegerSet: Integer,
    IndicesSet: Indices,
    Indices: IntegerTuple,
    Object: Cell
}

parent_mapper = {
    Object: Objects,
    Indices: IndicesSet,
    Integer: IntegerSet,
    IntegerTuple: Indices,
    Cell: Object
}
