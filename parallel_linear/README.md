# Parallel Linears and Mixture of Experts

## update 2022.07.19
The `MoE.forward()` is now a standard Mixture of Experts (FFD). The mixture of attention code is suppose to use `MoE.map()` and `MoE.reduce()` functions.

## Mixture of Experts
Mixture of Experts (MoE) is a map-reduce style function.
The forward function maps different inputs to different experts. The reduce function sums these intermediate result together for each inputs. 
Parameter:
1. `input_size` - the size of input hidden states
2. `output_size` - the size of intermediate hidden states
3. `num_experts` - the number of total experts
4. `k` - the number of topk selected experts for each input
5. `cvloss`, `switchloss`, `zloss` - different load balancing losses.
6. `activation` - the activation function for intermediate states of MoE (FFD).

To install the classs:
```
pip3 install .
```
or
```
python3 setup.py install
```

To use the class:
```
from parallel_experts import MoE

moe = MoE()
```
The `MoE` is map-reduce still function. To use the function, first map the input `x` with the map function:
```
mapped = moe.map(x)
```
Then you can pass the mapped and projected output through attention (for mixture of attention) or non-linear activation (for mixture of FFD) to get the processed matrix `y`
Lastly, you feed `y` to the reduce function to get the output of mixture of attention/FFD.
```
output = moe.reduce(y)
```

## Parallel Linears
Parallel linears is a part of MoE. 
Input to the function includes:
1. Input matrix $ X $, a $ B \times D_{in} $ matrix, where $B$ is the total number of input vectors.
2. Weight matrix $ W $, a $ N \times D_{out} \times D_{in} $ matrix, where $N$ is the number of linear kernels.
3. Routing vector $ R $, a $ B $ dimensional vector, where each elements $ R_i $ ( $ 0 \leq R_i < N $ ) is the index of weight matrix for $ i $-th input vector. The input matrix and routing vector are sorted according to the weight index. For example, a valid routing vector is $ [0\ 0\ 0\ 1\ 1\ 2\ 3\ 3\ 3\ 3] $
4. Start indices vector $ S $, a $ B $ dimensional vector, where each elements $ S_i $ is the starting index for inputs of $ i $-th weight matrix.
5. End indices vector $ E $, a $ B $ dimensional vector, where each elements $ E_i $ is the ending index for inputs of $ i $-th weight matrix.

The output of the function is the $ \left[ \begin{matrix} W_{R_1} X_1, W_{R_2} X_2, ..., W_{R_B} X_B \end{matrix} \right] $.

To run test:
```
python test.py
```