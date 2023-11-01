#include <torch/extension.h>

void phf_launcher(torch::Tensor points,
                  torch::Tensor masks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("phf", &phf_launcher);
}