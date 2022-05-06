#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

template <typename scalar_t>
__global__ void maxmin_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    size_t outer_size,
    size_t axis_length,
    size_t inner_stride,
    int32_t group_size,
    scalar_t* __restrict__ argsort,
    scalar_t* __restrict__ output) {
  const int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int axis_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_idx = blockIdx.z * blockDim.z + threadIdx.z;
  const int inner_idx = group_size * axis_idx;
  if (outer_idx < outer_size && stride_idx < inner_stride) {
    
    const int start_idx = inner_stride * axis_length * outer_idx + inner_idx * inner_stride + stride_idx;
    
    if (inner_idx < axis_length - 1) { // TODO: FIX CONDITION FOR WHAT HAPPENS AT ENDS OF AXIS (maybe just fix input to always be a multiple?)

      // copy input to output and modify output in-place
      for (int i = 0; i < group_size; i++) {
        output[start_idx + inner_stride * i] = input[start_idx + inner_stride * i];
        argsort[start_idx + inner_stride * i] = i;
      }

      // insertion sort
      for (int i = 1; i < group_size; i++) {
        scalar_t key = output[start_idx + inner_stride * i];
        int j = i - 1;
        while (j >= 0 && output[start_idx + inner_stride * j] > key) {
          output[start_idx + inner_stride * (j + 1)] = output[start_idx + inner_stride * j];
          argsort[start_idx + inner_stride * (j + 1)] = argsort[start_idx + inner_stride * j];
          j = j - 1;
        }
        output[start_idx + inner_stride * (j + 1)] = key;
        argsort[start_idx + inner_stride * (j + 1)] = i;
      }

    } else if (inner_idx < axis_length) {
      // In range, but at end of sorting axis
      output[start_idx] = input[start_idx];
    }

  }

/*
  // REGULAR C++ INSERTION SORT CODE
  int i, key, j;
  for (i = 1; i < n; i++) {
      key = arr[i];
      j = i - 1;

        // Move elements of arr[0..i-1], that are
        // greater than key, to one position ahead
        // of their current position 
      while (j >= 0 && arr[j] > key) {
          arr[j + 1] = arr[j];
          j = j - 1;
      }
      arr[j + 1] = key;
  }
*/

 


}

std::vector<at::Tensor> maxmin_cuda_forward(
    at::Tensor input,
    int32_t axis, 
    int32_t group_size) {
  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  };
  int inner_stride = 1;
  for (int i = true_axis + 1; i < num_dims; i++) {
    inner_stride *= input.size(i);
  }

  dim3 block(8, 8, 8);
  dim3 grid((outer_size + 7) / 8, (axis_length + 15) / 16, (inner_stride + 7) / 8);

  auto output = at::zeros_like(input);
  auto argsort = at::zeros_like(input);
  AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
    maxmin_cuda_forward_kernel<scalar_t><<<grid, block>>>(
        input.data<scalar_t>(),
        outer_size,
        axis_length,
        inner_stride,
        group_size,
        argsort.data<scalar_t>(),
        output.data<scalar_t>());
  }));
  std::vector<at::Tensor> outputs;
  outputs.push_back(output);
  outputs.push_back(argsort);
  return outputs;
}



template <typename scalar_t>
__global__ void maxmin_cuda_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ argsort,
    size_t outer_size,
    size_t axis_length,
    size_t inner_stride,
    int32_t group_size,
    scalar_t* __restrict__ output_grad) {
  const int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int axis_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_idx = blockIdx.z * blockDim.z + threadIdx.z;
  const int inner_idx = group_size * axis_idx;

  if (outer_idx < outer_size && stride_idx < inner_stride) {
    const int start_idx = inner_stride * axis_length * outer_idx + inner_idx * inner_stride + stride_idx;
    if (inner_idx < axis_length - 1) { 

      // use sorted indices to undo the sort on the gradient
      for (int i = 0; i < group_size; i++) {
        int original_i = argsort[start_idx + inner_stride * i];
        output_grad[start_idx + inner_stride * original_i] = grad[start_idx + inner_stride * i];
      }

      
    } else if (inner_idx < axis_length) {
      // In range, but at end of sorting axis
      output_grad[start_idx] = grad[start_idx];
    }

  }

}

at::Tensor maxmin_cuda_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis,
    int32_t group_size,
    at::Tensor argsort) {
  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  };
  int inner_stride = 1;
  for (int i = true_axis + 1; i < num_dims; i++) {
    inner_stride *= input.size(i);
  }

  dim3 block(8, 8, 8);
  dim3 grid((outer_size + 7) / 8, (axis_length + 15) / 16, (inner_stride + 7) / 8);

  auto output_grad = at::zeros_like(grad);

  AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_backward_cuda", ([&] {
    maxmin_cuda_backward_kernel<scalar_t><<<grid, block>>>(
        input.data<scalar_t>(),
        grad.data<scalar_t>(),
        argsort.data<scalar_t>(),
        outer_size,
        axis_length,
        inner_stride,
        group_size,
        output_grad.data<scalar_t>());
  }));
  return output_grad;
}

