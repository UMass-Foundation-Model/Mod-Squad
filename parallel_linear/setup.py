from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='parallel_experts',
      packages=find_packages(), 
      # ext_modules=[cpp_extension.CUDAExtension('parallel_linear',
      #                 ['parallel_linear.cc', 
      #                 'parallel_linear_kernel.cu'
      #                 ])],
      # cmdclass={'build_ext': cpp_extension.BuildExtension},
      install_requires=[
            'torch'
      ])