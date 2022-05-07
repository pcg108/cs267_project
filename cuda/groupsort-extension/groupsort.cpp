#include <torch/torch.h>

#include <vector>

// CUDA declarations
std::vector<at::Tensor> groupsort_cuda_forward(
    at::Tensor input,
    int32_t axis, 
    int32_t group_size);

at::Tensor groupsort_cuda_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis,
    int32_t group_size,
    at::Tensor argsort);


// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> groupsort_forward(
    at::Tensor input,
    int32_t axis, 
    int32_t group_size) {
  CHECK_INPUT(input);

  return groupsort_cuda_forward(input, axis, group_size);
  
}

at::Tensor groupsort_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis,
    int32_t group_size,
    at::Tensor argsort) {
  CHECK_INPUT(input);
  CHECK_INPUT(grad);

  return groupsort_cuda_backward(input, grad, axis, group_size, argsort);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &groupsort_forward, "GroupSort forward (CUDA)");
  m.def("backward", &groupsort_backward, "GroupSort backward (CUDA)");
}
