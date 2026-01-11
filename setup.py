from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension#, include_paths as torch_include_paths, library_paths as torch_library_paths
import os, sys

#here = os.path.dirname(os.path.abspath(__file__))
#
## Prefer the active conda env; fall back to CUDA_HOME if set; finally /usr/local/cuda
#CUDA_HOME = os.environ.get("CONDA_PREFIX") or os.environ.get("CUDA_HOME", "/usr/local/cuda")
#
#cuda_includes = [
#    os.path.join(CUDA_HOME, "targets/x86_64-linux/include"),
#    os.path.join(CUDA_HOME, "targets/x86_64-linux/include/cccl"),  # needed for <cuda/std/...> on CUDA 12.x
#]
#third_party_inc = [os.path.join(here, "third_party", "glm")]
#
## Sanity check: make sure the critical headers are where we think they are
#must_exist = [
#    os.path.join(cuda_includes[0], "cuda.h"),
#    os.path.join(cuda_includes[1], "cuda", "std", "type_traits"),
#    os.path.join(third_party_inc[0]),  # glm dir
#]
#missing = [p for p in must_exist if not os.path.exists(p)]
#if missing:
#    sys.stderr.write("Missing required include paths/files:\n  - " + "\n  - ".join(missing) + "\n")
#    sys.stderr.write("Tip: export CUDA_HOME to your conda env, e.g. `export CUDA_HOME=$CONDA_PREFIX`.\n")
#    sys.exit(1)
# third_party_inc = ["third_party", "glm"]

setup(
    name="EV3DGS",
    packages=["ev3dgs"],
    ext_modules=[
        CUDAExtension(
            name="ev3dgs._C",
            sources=[
                "cuda_ev3dgs/rasterizer_impl.cu",
                "cuda_ev3dgs/forward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            # include_dirs= third_party_inc, # + torch_include_paths(),
            #library_dirs=torch_library_paths(),
            define_macros=[("_GLIBCXX_USE_CXX11_ABI", "1")],
            # include_dirs=[os.path.join(os.path.dirname(__file__), "third_party")],
            extra_compile_args={
                "cxx":  ["-std=c++17"],
                "nvcc": ["-std=c++17", "-Xcompiler", "-fno-gnu-unique", "-D_GLIBCXX_USE_CXX11_ABI=1", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
