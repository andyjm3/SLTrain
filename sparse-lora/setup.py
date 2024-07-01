import os
from setuptools import setup, Extension

from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

extension = CppExtension(
    'sltrain_linear',
    ['sparse_linear.cpp'],
)

setup(
    name='sparse_lowrank',
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension},
)