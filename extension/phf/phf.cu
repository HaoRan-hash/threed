#include "phf.h"
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>

#define THREAD_PER_BLOCK 256

__global__ void phf_kernel(torch::PackedTensorAccessor32<float, 4> points,
                           torch::PackedTensorAccessor32<float, 4> masks)
{
    int b = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= points.size(1))
        return;

    for (int k = 0; k < points.size(2); k++)
    {
        int x = points[b][tid][k][0] > 0 ? 4 : 0;
        int y = points[b][tid][k][1] > 0 ? 2 : 0;
        int z = points[b][tid][k][2] > 0 ? 1 : 0;

        int temp = x + y + z;
        masks[b][tid][k][temp] = 1.0;
    }
}

void phf_launcher(torch::Tensor points,
                  torch::Tensor masks)
{
    const at::cuda::OptionalCUDAGuard device_guard(points.device());

    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp = points.size(1) / THREAD_PER_BLOCK + ((points.size(1) % THREAD_PER_BLOCK) > 0);
    dim3 block(temp, points.size(0), 1);

    // 启动kernel
    phf_kernel<<<block, thread>>> (points.packed_accessor32<float, 4>(),
                                   masks.packed_accessor32<float, 4>());
}