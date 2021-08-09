#include <torch/extension.h>


#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


// cuda launcher define
void map_count_launcher(const at::Tensor &coords,
                        const int N, const int B,
                        const int H, const int W,
                        at::Tensor &out_counts);

void point2range_launcher(const at::Tensor &points,
                          const int N, const int B, const int C,
                          const int H, const int W,
                          const int *coords_,
                          const int *counts_,
                          at::Tensor &rv_image);

void point2range_backward_launcher(const at::Tensor &grad_out,
                                   const int N, const int B, const int C,
                                   const int H, const int W,
                                   const int *coords_,
                                   const int *counts_,
                                   at::Tensor &grad_in);

// bind func
void map_count_gpu(const at::Tensor &coords,
                   at::Tensor &out_counts) {
    // coords: Nx3, [bs_idx, x, y]
    // out_counts: BxHxW

    CHECK_INPUT(coords);
//    at::DeviceGuard guard(coords.device());
    int N = coords.size(0);
    int B = out_counts.size(0);
    int H = out_counts.size(1);
    int W = out_counts.size(2);
    return map_count_launcher(coords, N, B, H, W, out_counts);
}

void point2range_gpu(const at::Tensor &points,
                     const at::Tensor &coords,
                     const at::Tensor &counts,
                     at::Tensor &rv_image) {
    // points NxC <type template>
    // coords Nx3, [bs_idx, x, y] int
    // counts BxHxW int
    // rv_image BxCxHxW <type template>

    CHECK_INPUT(points);
    CHECK_INPUT(coords);
    CHECK_INPUT(counts);

    int N = points.size(0);
    int B = rv_image.size(0);
    int C = rv_image.size(1);
    int H = rv_image.size(2);
    int W = rv_image.size(3);

    const int *coords_ = coords.data<int>();
    const int *counts_ = counts.data<int>();

    return point2range_launcher(points, N, B, C, H, W, coords_, counts_, rv_image);
}

void point2range_gpu_backward(const at::Tensor &grad_out,
                              const at::Tensor &coords,
                              const at::Tensor &counts,
                              at::Tensor &grad_in) {
    // grad_out BxCxHxW <type template> grad of rv_image
    // coords Nx3 int
    // counts BxHxW int
    // grad_in NxC <type template> grad of input points

    CHECK_INPUT(grad_out);
    CHECK_INPUT(coords);
    CHECK_INPUT(counts);

    int N = coords.size(0);
    int B = grad_out.size(0);
    int C = grad_out.size(1);
    int H = grad_out.size(2);
    int W = grad_out.size(3);

    const int *coords_ = coords.data<int>();
    const int *counts_ = counts.data<int>();

    return point2range_backward_launcher(grad_out, N, B, C, H, W, coords_, counts_, grad_in);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("map_count", &map_count_gpu, "count points number in each range image pixel (CUDA)");
m.def("point2range", &point2range_gpu, "project points to range by MeanVFE forward (CUDA)");
m.def("point2range_backward", &point2range_gpu_backward, "project points to range by MeanVFE backward (CUDA)");
}
