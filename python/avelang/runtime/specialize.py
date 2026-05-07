from .._utils import canonicalize_dtype
from .jit import JITCallable
from ..language import constexpr


# Type specialization in Python
# TODO: Adopt the upstream specialization optimization pattern.
def reference_specialize_impl(backend, arg, is_const, specialize_value, align):
    if arg is None:
        return ("constexpr", None)
    elif isinstance(arg, bool):
        return ("u1", None)
    elif isinstance(arg, int):
        key = backend.get_int_specialization(arg, align=align) if specialize_value else None
        if arg == 1 and specialize_value:
            return ("constexpr", 1)
        elif -(2**31) <= arg and arg <= 2**31 - 1:
            return ("i32", key)
        elif 2**63 <= arg and arg <= 2**64 - 1:
            return ("u64", key)
        else:
            return ("i64", key)
    elif isinstance(arg, float):
        return ("fp32", None)
    elif hasattr(arg, "data_ptr"):
        dsk = (arg.dtype, is_const)
        res = ("*k" if dsk[1] else "*") + canonicalize_dtype(dsk[0])
        key = backend.get_tensor_specialization(arg, align=align) if specialize_value else None
        return (res, key)
    elif isinstance(arg, JITCallable):
        return ("constexpr", arg.cache_key)
    elif isinstance(arg, constexpr):
        return ("constexpr", arg)
    elif isinstance(arg, tuple):
        spec = [reference_specialize_impl(backend, x, False, True, True) for x in arg]

        def make_tuple(vals):
            return type(arg)(*vals) if hasattr(arg, "_fields") else tuple(vals)

        tys = make_tuple([x[0] for x in spec])
        keys = make_tuple([x[1] for x in spec])
        return (tys, keys)
    else:
        raise TypeError("Unsupported type: %s" % type(arg))
