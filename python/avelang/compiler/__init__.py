__all__ = ["compile", "make_backend", "ASTSource", "CompiledKernel"]


def __getattr__(name):
    if name in __all__:
        from . import compiler as _compiler

        return getattr(_compiler, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
