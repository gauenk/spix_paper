
import os

# Set environment variables to use ccache
os.environ['PYTORCH_NVCC'] = "ccache nvcc"
os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '1'

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="spix_paper",
    py_modules=["spix_paper"],
    install_requires=[],
    package_dir={"": "."},
    packages=find_packages("."),
    package_data={'': ['*.so']},
    include_package_data=True,
    ext_modules=[
        CUDAExtension('spix_paper_cuda', [
            # -- pairwise distance --
            'spix_paper/csrc/pwd/pair_wise_distance_cuda_source.cu',
            # -- apis --
            'spix_paper/csrc/sna/attn_reweight.cu',
            'spix_paper/csrc/sna/gather_sims.cu',
            # -- pybind --
            "spix_paper/csrc/pybind.cpp",
        ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-O3','-w']}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
