#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>


#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}


__global__ void map_count_cuda(const int *coords_,
                               const int N, const int H, const int W,
                               int *out_counts_) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;          // compute absolute index
    if (index >= N) {
        return;
    }
    int bs = coords_[3 * index];                                              // find u, v coords
    int u = coords_[3 * index + 1];                                              // find u, v coords
    int v = coords_[3 * index + 2];                                              // find u, v coords
    atomicAdd(&(out_counts_[bs * H * W + v * W + u]), 1);
}

template<typename scalar_t>
__global__ void point2range_cuda(const scalar_t *points_,
                                 const int parallel_size,
                                 const int C, const int H, const int W,
                                 const int *coords_,
                                 const int *counts_,
                                 scalar_t *rv_image) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;          // compute absolute index
    if (index >= parallel_size) {
        return;
    }
    const scalar_t value = points_[index];
    const int pts_num = index / C;
    const int channel_num = index % C;

    int bs = coords_[3 * pts_num];
    int u = coords_[3 * pts_num + 1];
    int v = coords_[3 * pts_num + 2];

    const int count = counts_[bs * H * W + v * W + u];
    if (count == 0) { // theoretically it cannot be 0!
        return;
    }
    int rv_pos = bs * (C * H * W) + channel_num * (H * W) + v * W + u;
    atomicAdd(&(rv_image[rv_pos]), value / float(count));
}

template<typename scalar_t>
__global__ void point2range_backward_cuda(const scalar_t *grad_out_,
                                          const int parallel_size,
                                          const int C, const int H, const int W,
                                          const int *coords_,
                                          const int *counts_,
                                          scalar_t *grad_in_) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;          // compute absolute index
    if (index >= parallel_size) {
        return;
    }
    const int pts_num = index / C;
    const int channel_num = index % C;

    int bs = coords_[3 * pts_num];
    int u = coords_[3 * pts_num + 1];
    int v = coords_[3 * pts_num + 2];

    const int count = counts_[bs * H * W + v * W + u];
    if (count == 0) {// theoretically it cannot be 0!
        return;
    }
    int rv_pos = bs * (C * H * W) + channel_num * (H * W) + v * W + u;

    grad_in_[index] = grad_out_[rv_pos] / float(count);
}


void map_count_launcher(const at::Tensor &coords,
                        const int N, const int B,
                        const int H, const int W,
                        at::Tensor &out_counts) {

    const int *coords_ = coords.data<int>();
    int *out_counts_ = out_counts.data<int>();

    map_count_cuda << < GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream() >> > (
            coords_, N, H, W, out_counts_
    );
    THCudaCheck(cudaGetLastError());
}


void point2range_launcher(const at::Tensor &points,
                          const int N, const int B, const int C,
                          const int H, const int W,
                          const int *coords_,
                          const int *counts_,
                          at::Tensor &rv_image) {

    const int parallel_size = N * C;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "point2range_launcher", ([&] {
        const scalar_t *points_ = points.data<scalar_t>();
        scalar_t *rv_image_ = rv_image.data<scalar_t>();

        point2range_cuda<scalar_t>
                << < GET_BLOCKS(parallel_size), THREADS_PER_BLOCK,
                0, at::cuda::getCurrentCUDAStream() >> > (
                points_, parallel_size, C, H, W, coords_, counts_, rv_image_
        );
    }));
    THCudaCheck(cudaGetLastError());
}


void point2range_backward_launcher(const at::Tensor &grad_out,
                                   const int N, const int B, const int C,
                                   const int H, const int W,
                                   const int *coords_,
                                   const int *counts_,
                                   at::Tensor &grad_in) {

    const int parallel_size = N * C;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "point2range_backward_launcher", ([&] {
        const scalar_t *grad_out_ = grad_out.data<scalar_t>();
        scalar_t *grad_in_ = grad_in.data<scalar_t>();

        point2range_backward_cuda<scalar_t>
                << < GET_BLOCKS(parallel_size), THREADS_PER_BLOCK,
                0, at::cuda::getCurrentCUDAStream() >> > (
                grad_out_, parallel_size, C, H, W, coords_, counts_, grad_in_
        );
    }));
    THCudaCheck(cudaGetLastError());
}