#include "points_group.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <stdio.h>

#define THREAD_PER_BLOCK 256

__global__ void neighbor_group_kernel(torch::PackedTensorAccessor32<float, 3> points,
                                      torch::PackedTensorAccessor32<int64_t, 3> indices,
                                      torch::PackedTensorAccessor32<float, 4> group_output)
{
    int neighbor_num = indices.size(2);
    int b = blockIdx.z;
    int c = blockIdx.y;
    int m = (blockIdx.x * blockDim.x + threadIdx.x) / neighbor_num;
    int k = (blockIdx.x * blockDim.x + threadIdx.x) % neighbor_num;
    if (m >= indices.size(1))
        return;
    
    group_output[b][c][k][m] = points[b][c][indices[b][m][k]];
}

void neighbor_group_launcher(torch::Tensor points,
                             torch::Tensor indices,
                             torch::Tensor group_output)
{
    const at::cuda::OptionalCUDAGuard device_guard(points.device());

    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp1 = group_output.size(2) * group_output.size(3);
    int temp2 = temp1 / THREAD_PER_BLOCK + ((temp1 % THREAD_PER_BLOCK) > 0);
    dim3 block(temp2, group_output.size(1), group_output.size(0));

    // 启动kernel
    neighbor_group_kernel<<<block, thread>>> (points.packed_accessor32<float, 3>(),
                                              indices.packed_accessor32<int64_t, 3>(),
                                              group_output.packed_accessor32<float, 4>());
}

__global__ void neighbor_group_grad_kernel(torch::PackedTensorAccessor32<float, 4> output_grad,
                                           torch::PackedTensorAccessor32<int64_t, 3> indices,
                                           torch::PackedTensorAccessor32<float, 3> points_grad)
{
    int neighbor_num = indices.size(2);
    int b = blockIdx.z;
    int c = blockIdx.y;
    int m = (blockIdx.x * blockDim.x + threadIdx.x) / neighbor_num;
    int k = (blockIdx.x * blockDim.x + threadIdx.x) % neighbor_num;
    if (m >= indices.size(1))
        return;

    atomicAdd(&points_grad[b][c][indices[b][m][k]], output_grad[b][c][k][m]);
}

void neighbor_group_grad_launcher(torch::Tensor output_grad,
                                  torch::Tensor indices,
                                  torch::Tensor points_grad)
{
    const at::cuda::OptionalCUDAGuard device_guard(output_grad.device());

    // 配置block和thread
    dim3 thread(THREAD_PER_BLOCK, 1, 1);
    int temp1 = output_grad.size(2) * output_grad.size(3);
    int temp2 = temp1 / THREAD_PER_BLOCK + ((temp1 % THREAD_PER_BLOCK) > 0);
    dim3 block(temp2, output_grad.size(1), output_grad.size(0));

    // 启动kernel
    neighbor_group_grad_kernel<<<block, thread>>> (output_grad.packed_accessor32<float, 4>(),
                                                   indices.packed_accessor32<int64_t, 3>(),
                                                   points_grad.packed_accessor32<float, 3>());
}