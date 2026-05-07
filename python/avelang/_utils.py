from __future__ import annotations

from functools import reduce
from typing import Any, Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .language import core

    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def is_iterable(x):
    # TODO: Support the tuple type from the language
    return isinstance(x, (list, tuple))


def find_paths_if(iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]) -> list[ObjPath]:
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


type_canonicalisation_dict = {
    # we canonicalise all bools to be unsigned:
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    # floating-point dtypes:
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    # signed integers:
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    # unsigned integers:
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "u8": "u8",
    "u16": "u16",
    "u32": "u32",
    "u64": "u64",
    "void": "void",
}

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


def canonicalize_dtype(dtype):
    dtype_str = str(dtype).split(".")[-1]
    return type_canonicalisation_dict[dtype_str]


def is_namedtuple(val):
    return isinstance(val, type) and issubclass(val, tuple) and hasattr(val, "_fields")
