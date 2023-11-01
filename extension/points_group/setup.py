from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='points_group',
    ext_modules=[CUDAExtension('points_group', 
                               ['points_group.cu'])],
    cmdclass={'build_ext': BuildExtension}
)