#include <torch/extension.h>

void neighbor_group_launcher(torch::Tensor points,
                             torch::Tensor indices,
                             torch::Tensor group_output);

void neighbor_group_grad_launcher(torch::Tensor output_grad,
                                  torch::Tensor indices,
                                  torch::Tensor points_grad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("neighbor_group", &neighbor_group_launcher);
    m.def("neighbor_group_grad", &neighbor_group_grad_launcher);
}