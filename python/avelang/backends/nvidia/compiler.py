from dataclasses import dataclass

from ..compiler import BaseBackend, GPUTarget
from ...compiler.code_generator import compile_to_binary


@dataclass(frozen=True)
class NvidiaCompilerOptions:
    num_warps: int = -1


class NvidiaCompiler(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.tuple == "nvptx64-nvidia-cuda"

    def parse_options(self, options) -> object:
        if options is None:
            return NvidiaCompilerOptions()
        if isinstance(options, NvidiaCompilerOptions):
            return options

        args = {}
        if "num_warps" in options and options["num_warps"] is not None:
            args["num_warps"] = options["num_warps"]
        return NvidiaCompilerOptions(**args)

    def compile(self, src, target, options=None):
        if options is None:
            options = self.parse_options({})
        elif isinstance(options, dict):
            options = self.parse_options(options)
        return compile_to_binary(src, target, opt_level=2, options=options)
