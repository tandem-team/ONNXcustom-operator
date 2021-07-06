from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='rlcustom_operator',
      ext_modules=[cpp_extension.CppExtension('rlcustom_operator', ['rlcustom_operator.cpp'])],include_dirs = ["~/.local/lib/python3.6/site-packages/torch/share/cmake/Torch"],cmdclass={'build_ext': cpp_extension.BuildExtension})
