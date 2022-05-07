from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='groupsort',
      version='0.0.1',
      description='GroupSort activation function.',
      packages=['groupsort'],
      ext_modules=[CUDAExtension('groupsort.groupsort_extension', ['groupsort-extension/groupsort.cpp', 'groupsort-extension/groupsort_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})