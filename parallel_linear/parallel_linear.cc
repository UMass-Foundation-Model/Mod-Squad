#include <torch/extension.h>

torch::Tensor  parallel_linear_fwd_interface(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> parallel_linear_bwd_interface(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor parallel_linear_fwd(torch::Tensor input, torch::Tensor weight, torch::Tensor indices) {
    if(input.device().type() == torch::kCPU) {
        int bsz = input.size(0);
        torch::Tensor output = torch::zeros({bsz, weight.size(1)});
        for (int i = 0; i < bsz; ++i)
        {
            output[i] = torch::mv(weight[indices[i]], input[i]);
        }
        return output;
    } 
    else if (input.device().type() == torch::kCUDA){
        CHECK_INPUT(input);
        CHECK_INPUT(weight);
        CHECK_INPUT(indices);
        TORCH_CHECK(indices.dtype() == torch::kInt64,
            "Indices Datatype not implemented");

        return parallel_linear_fwd_interface(input, weight, indices);
    }
    AT_ERROR("No such device: ", input.device());
}

std::vector<torch::Tensor> parallel_linear_bwd(torch::Tensor grad_out, torch::Tensor input, torch::Tensor weight, 
                                            torch::Tensor indices, torch::Tensor start_indices, torch::Tensor end_indices) {
    if(input.device().type() == torch::kCPU) {
        int bsz = input.size(0);
        torch::Tensor d_input = torch::zeros_like(input);
        torch::Tensor d_weight = torch::zeros_like(weight);

        for (int i = 0; i < bsz; ++i)
        {
            d_input[i] = torch::mv(weight[indices[i]].transpose(0, 1), grad_out[i]);
            d_weight[indices[i]] += torch::outer(grad_out[i], input[i]);
        }
        return {d_input, d_weight};
    } 
    else if (input.device().type() == torch::kCUDA){
        CHECK_INPUT(input);
        CHECK_INPUT(weight);
        CHECK_INPUT(indices);
        CHECK_INPUT(grad_out);
        TORCH_CHECK(indices.dtype() == torch::kInt64,
            "Indices Datatype not implemented");
        
        return parallel_linear_bwd_interface(grad_out, input, weight, indices, start_indices, end_indices);
    }
    AT_ERROR("No such device: ", input.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &parallel_linear_fwd, "Parallel linear forward");
  m.def("backward", &parallel_linear_bwd, "Parallel linear backward");
}