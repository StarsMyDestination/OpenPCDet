import os
import sys
import subprocess

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools.command.install import install


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


# def make_cuda_ext(name, module, sources):
#     cuda_ext = CUDAExtension(
#         name='%s.%s' % (module, name),
#         sources=[os.path.join(*module.split('.'), src) for src in sources]
#     )
#     return cuda_ext


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


class PostInstallation(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Note: buggy for kornia==0.5.3 and it will be fixed in the next version.
        # Set kornia to 0.5.2 temporarily
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'kornia==0.5.2', '--no-dependencies'])


if __name__ == '__main__':
    version = '0.3.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='pcdet',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'torch>=1.1',
            # 'spconv',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
            'install': PostInstallation,
            # Post installation cannot be done. ref: https://github.com/pypa/setuptools/issues/1936.
            # 'develop': PostInstallation,
        },
        ext_modules=[
            make_cuda_ext(
                name='sparse_conv_ext',
                module='pcdet.ops.spconv',
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*'pcdet.ops.spconv'.split('.'),
                                     'include/'))
                ],
                sources=[
                    'src/all.cc',
                    'src/reordering.cc',
                    'src/reordering_cuda.cu',
                    'src/indice.cc',
                    'src/indice_cuda.cu',
                    'src/maxpool.cc',
                    'src/maxpool_cuda.cu',
                ],
                extra_args=['-w', '-std=c++14']),

            make_cuda_ext(
                name='voxel_layer',
                module='pcdet.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ]),

            make_cuda_ext(
                name='rangeview_cuda',
                module='pcdet.ops.rangeview',
                sources=[
                    'src/point_to_range.cpp',
                    'src/point_to_range_cuda.cu',
                ]),

            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp',
                    'src/voxel_query_gpu.cu',
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
        ],
    )
