#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

const int BLOCK_SIZE = 1024;
const int BLOCK_WIDTH = 32;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename scalar_t>
__global__ void parallel_linear_fwd_cuda(scalar_t* input, float* weight, int64_t* indices, scalar_t* ret,
                              int input_size, int output_size) {
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int i = blockIdx.y * blockDim.x + tx;

    const int index = indices[b];
    float ret_i = 0;

    for (int j_start = 0 ; j_start < input_size ; j_start += blockDim.x) {
        __shared__ scalar_t xs[BLOCK_SIZE];
        
        if ((j_start + tx) < input_size) {
            xs[tx] = input[b * input_size + j_start + tx];
        } else {
            xs[tx] = 0;
        }
        __syncthreads();

        if (i < output_size) {
            for(int j = 0 ; (j < blockDim.x) && (j_start + j < input_size) ; ++j) {
                float w = weight[index * input_size * output_size + (j_start + j) * output_size + i];
                ret_i += w * xs[j];
            }
        }
        __syncthreads();
    }

    if (i < output_size) {
        ret[b * output_size + i] = (scalar_t)ret_i;
    }
}


template <typename scalar_t>
__global__ void parallel_linear_weight_bwd_cuda(scalar_t* grad_out, scalar_t* input, 
                                        int64_t* start_indices, int64_t* end_indices, float* d_weight, 
                                        int input_size, int output_size) {
    int expert_index = blockIdx.x;
    int input_starter = blockIdx.y * BLOCK_WIDTH;
    int output_starter = blockIdx.z * BLOCK_WIDTH;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = input_starter + tx;
    int j = output_starter + ty;
    int start_index = start_indices[expert_index];
    int end_index = end_indices[expert_index];

    float d_weight_i = 0;

    __shared__ scalar_t inputs[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ scalar_t grads[BLOCK_WIDTH][BLOCK_WIDTH];

    for (int b_starter = start_index; b_starter < end_index ; b_starter += BLOCK_WIDTH) {
        if (b_starter + tx < end_index) {
            if (input_starter + ty < input_size) {
                inputs[tx][ty] = input[(b_starter + tx) * input_size + input_starter + ty];
            } else {
                inputs[tx][ty] = 0;
            }
            if (output_starter + ty < output_size) {
                grads[tx][ty] = grad_out[(b_starter + tx) * output_size + output_starter + ty];
            } else {
                grads[tx][ty] = 0;
            }
        } else {
            inputs[tx][ty] = 0;
            grads[tx][ty] = 0;
        }
        __syncthreads();

        for (int b = 0 ; b < BLOCK_WIDTH; ++b) {
            d_weight_i += inputs[b][tx] * grads[b][ty];
        }
        __syncthreads();
    }

    if ((i < input_size) && (j < output_size)) {
        int w_index = expert_index * input_size * output_size + i * output_size + j;
        d_weight[w_index] = d_weight_i;
    }
}


__host__ torch::Tensor parallel_linear_fwd_interface(torch::Tensor input, torch::Tensor weight, torch::Tensor indices) {
    // nblock = 1;
    // printf("%d, %d", nblock, BLOCKSIZE);
    // gpuErrchk(cudaPeekAtLastError());

    const int bsz = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weight.size(2);

    auto options =torch::TensorOptions()
                        .dtype(input.dtype())
                        .layout(torch::kStrided)
                        .device(input.device());
    torch::Tensor output = torch::zeros({bsz, output_size}, options);

    const int threads = min(BLOCK_SIZE, output_size);
    const dim3 blocks(bsz, (output_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "parallel_linear_fwd_cuda", ([&] {
        parallel_linear_fwd_cuda<<<blocks, threads>>>(
            input.data<scalar_t>(), 
            weight.data<float>(), 
            indices.data<int64_t>(), 
            output.data<scalar_t>(), 
            input_size, 
            output_size);
    }));

    gpuErrchk(cudaPeekAtLastError());

    return output;
}

__host__ std::vector<torch::Tensor> parallel_linear_bwd_interface(
    torch::Tensor grad_out, torch::Tensor input, torch::Tensor weight, 
    torch::Tensor indices, torch::Tensor start_indices, torch::Tensor end_indices) {
    // nblock = 1;
    // printf("%d, %d", nblock, BLOCKSIZE);
    // gpuErrchk(cudaPeekAtLastError());

    const int bsz = input.size(0);
    const int input_size = input.size(1);
    const int output_size = grad_out.size(1);
    const int n_experts = weight.size(0);

    torch::Tensor d_input = torch::zeros_like(input);
    torch::Tensor d_weight = torch::zeros_like(weight);

    torch::Tensor weight_t = weight.transpose(1,2).contiguous();
    
    const int threads = min(BLOCK_SIZE, input_size);
    const dim3 blocks(bsz, (input_size + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "parallel_linear_input_bwd_cuda", ([&] {
        parallel_linear_fwd_cuda<<<blocks, threads>>>(
            grad_out.data<scalar_t>(), 
            weight_t.data<float>(), 
            indices.data<int64_t>(), 
            d_input.data<scalar_t>(), 
            output_size,
            input_size);
    }));

    const dim3 weight_threads(BLOCK_WIDTH, BLOCK_WIDTH);
    const dim3 dimGrid_weight(n_experts, (input_size + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (output_size + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "parallel_linear_weight_bwd_cuda", ([&] {
        parallel_linear_weight_bwd_cuda<<<dimGrid_weight, weight_threads>>>(
            grad_out.data<scalar_t>(), 
            input.data<scalar_t>(), 
            start_indices.data<int64_t>(), 
            end_indices.data<int64_t>(), 
            d_weight.data<float>(), 
            input_size, 
            output_size);
    }));
    
    gpuErrchk(cudaPeekAtLastError());

    return {d_input, d_weight};
}