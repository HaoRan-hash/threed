from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='phf_cuda',
    ext_modules=[CUDAExtension('phf_cuda', 
                               ['phf.cu'])],
    cmdclass={'build_ext': BuildExtension}
)