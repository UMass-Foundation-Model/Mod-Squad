import time

import torch
import torch.nn.functional as F

from parallel_experts import ParallelLinear

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

NUM_EXPERTS=128
INPUT_SIZE=512
OUTPUT_SIZE=128
BSZ=512 * 32 * 8


def TorchParallelLinear(input, weight, bias, expert_size):
    output_list = []
    expert_size_list = expert_size.tolist()
    input_list = input.split(expert_size_list, dim=0)
    for i in range(NUM_EXPERTS):
        output_list.append(torch.mm(input_list[i], weight[i]) + bias[i])
    return torch.cat(output_list, dim=0)

    # output = torch.mm(input, weight[0])
    # return output


kernel_forward = 0
kernel_backward = 0
torch_forward = 0
torch_backward = 0
for t in range(200 + 1):
    weight = torch.rand((NUM_EXPERTS, INPUT_SIZE, OUTPUT_SIZE), requires_grad=True, device=cuda_device, dtype=torch.float16)
    bias = torch.rand((NUM_EXPERTS, OUTPUT_SIZE), requires_grad=True, device=cuda_device, dtype=torch.float16)
    input = torch.rand((BSZ, INPUT_SIZE), requires_grad=True, device=cuda_device, dtype=torch.float16)
    experts = torch.randint(NUM_EXPERTS, (BSZ,), device=cuda_device, dtype=torch.long)
    output_vector = torch.rand((BSZ, OUTPUT_SIZE), requires_grad=True, device=cuda_device, dtype=torch.float16)

    experts, _  = torch.sort(experts, dim=0)
    zeros = torch.zeros((BSZ, NUM_EXPERTS), device=cuda_device, dtype=torch.long)
    gates = zeros.scatter(1, experts[:, None], 1)
    expert_size = gates.sum(0)
    end_indices = expert_size.cumsum(0)
    start_indices = F.pad(end_indices[:-1], (1,0), value=0)

    torch.cuda.synchronize(cuda_device)

    start = time.time()
    function_output = ParallelLinear.apply(input, expert_size, weight, bias)
    function_output_sum = torch.einsum('bi,bi->b', function_output, output_vector).sum(0)
    torch.cuda.synchronize(cuda_device)
    forward_i = time.time() - start

    start = time.time()
    function_output_sum.backward()
    torch.cuda.synchronize(cuda_device)
    backward_i = time.time() - start

    if t > 0:
        kernel_forward += forward_i
        kernel_backward += backward_i
        print('Step {:2d} | K_Fwd: {:.3f} us | K_Bwd {:.3f} us'.format(t, forward_i * 1e6/1e5, backward_i * 1e6/1e5), end=' ')

    input_grad = input.grad
    weight_grad = weight.grad
    bias_grad = bias.grad

    input.grad = None
    weight.grad = None
    bias.grad = None

    torch.cuda.synchronize(cuda_device)

    start = time.time()
    output = TorchParallelLinear(input, weight, bias, expert_size)
    output_sum = torch.einsum('bi,bi->b', output, output_vector).sum(0)
    torch.cuda.synchronize(cuda_device)
    forward_i = time.time() - start

    start = time.time()
    output_sum.backward()
    torch.cuda.synchronize(cuda_device)
    backward_i = time.time() - start

    if t > 0:
        torch_forward += forward_i
        torch_backward += backward_i
        print('| T_Fwd: {:.3f} us | T_Bwd {:.3f} us'.format(forward_i * 1e6/1e5, backward_i * 1e6/1e5), end=' ')

    output_diff = torch.abs(output - function_output).max()
    input_grad_diff = torch.abs(input.grad - input_grad).max()
    weight_grad_diff = torch.abs(weight.grad - weight_grad).max()
    bias_grad_diff = torch.abs(bias.grad - bias_grad).max()

    if t > 0:
        print('| O_Diff: {:.3f} | Ig_Diff {:.3f} | Wg_Diff {:.3f} | bg_Diff {:.3f}'.format(
            output_diff, input_grad_diff, weight_grad_diff, bias_grad_diff))

    input.grad = None
    weight.grad = None

print('Kernel Forward: {:.3f} us | Kernel Backward {:.3f} us'.format(kernel_forward * 1e6/1e5, kernel_backward * 1e6/1e5))
print('Torch Forward: {:.3f} us | Torch Backward {:.3f} us'.format(torch_forward * 1e6/1e5, torch_backward * 1e6/1e5))
