from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

with open("requirements.txt") as f:
    required = f.read().splitlines()


extension = CppExtension(
    'sltrain_linear',
    ['sparse_linear.cpp'],
)

extensions = [extension]
cmdclass= {'build_ext': BuildExtension}

setup(
    name="sltrain",
    version="0.0",
    description="Sparse low-rank factorization",
    license="Apache 2.0",
    packages=["splora"],
    install_requires=required,
    ext_modules=extensions,
    cmdclass=cmdclass,
)