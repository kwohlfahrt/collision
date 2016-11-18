import pyopencl as cl

class Program:
    def __init__(self, ctx, options=None):
        with self.src.open("r") as f:
            self.program = cl.Program(ctx, f.read()).build(options or "")

        self.kernels = {name: getattr(self.program, name) for name in self.kernel_args}
        for name, kernel in self.kernels.items():
            kernel.set_scalar_arg_dtypes(self.kernel_args[name])

    @property
    def context(self):
        return self.program.get_info(cl.program_info.CONTEXT)
