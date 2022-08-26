# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        # sort experts
        self._expert_index, index_sorted_experts = torch.nonzero(gates)[:, 1].sort(0)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts, 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index[:, None])

    def dispatch(self, inp, k=1):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[torch.div(self._batch_index, k, rounding_mode='floor')].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, k=1, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(
            0) // k, expert_out[-1].size(1), dtype=stitched.dtype, requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(
            0, torch.div(self._batch_index, k, rounding_mode='floor'), stitched)
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, experts, k, dropout=0.1, concat=False, cvloss=0, switchloss=0.01, zloss=0.001):
        super(MoE, self).__init__()
        self.num_experts = len(experts)
        self.input_size = input_size
        self.experts = nn.ModuleList(experts)
        self.k = min(k, self.num_experts)
        self.concat = concat
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss

        self.f_gate = nn.Linear(input_size, self.num_experts, bias=False)
        nn.init.zeros_(self.f_gate.weight)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * F.normalize(freqs.sum(0), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def top_k_gating(self, x, skip_mask=None, sample_topk=0):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        logits = self.f_gate(x)
        # add noise
        # noise = torch.normal(torch.zeros_like(logits).float(), torch.ones_like(logits).float()/self.num_experts).cuda()
        # print('noise: ', noise.mean(), noise.min(), noise.max())
        # print('logits: ', logits.mean(), logits.min(), logits.max())
        logits = logits #+ noise

        if skip_mask:
            probs = torch.softmax(logits, dim=1) * skip_mask
        else:
            probs = torch.softmax(logits, dim=1)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6).detach()
        zeros = torch.zeros_like(probs, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        loss = 0
        loss += self.cvloss * self.compute_cvloss(probs) 
        loss += self.switchloss * self.compute_switchloss(probs, (gates > 0).float())
        loss += self.zloss * self.compute_zloss(logits)

        if self.concat:
            top_k_gates = top_k_gates.view(-1, 1)
            top_k_indices = top_k_indices.view(-1, 1)
            
            zeros = torch.zeros(top_k_gates.size(0), logits.size(1), device=x.device, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates, loss

    def forward(self, x, multiply_by_gates=True, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.view(-1, emb_size)
        if skip_mask:
            skip_mask = skip_mask.view(-1, 1)
        gates, loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        self.dispatcher = SparseDispatcher(gates)
        # if self.concat:
        #     x = x.repeat_interleave(self.k, 0)
        if self.concat:
            expert_inputs = self.dispatcher.dispatch(x, self.k)
        else:
            expert_inputs = self.dispatcher.dispatch(x)

        expert_outputs = [self.experts[i](
            expert_inputs[i]) for i in range(self.num_experts)]
        y = self.dispatcher.combine(expert_outputs, multiply_by_gates=multiply_by_gates)
        if self.concat:
            y = y.view(bsz, length, self.k, -1)
        else:
            y = y.view(bsz, length, -1)
        return y, loss

    def dispatch(self, x, experts, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = self.dispatcher.dispatch(x)
        expert_outputs = [experts[i](
            expert_inputs[i]) for i in range(self.num_experts)]
        y = self.dispatcher.combine(expert_outputs, k=self.k, multiply_by_gates=multiply_by_gates)
        y = y.view(bsz, length, -1)
        return y

class cvMoE(MoE):

    def __init__(self, input_size, experts, k, dropout=0.1, concat=False, cvloss=0, switchloss=0.01, zloss=0.001):
        super(cvMoE, self).__init__(input_size, experts, k, dropout=dropout, concat=concat, cvloss=0, switchloss=0., zloss=0.0)

    def compute_importance_loss(self, probs):
        assert probs.dim()==2
        probs = probs.sum(0) # [E]
        _std = torch.std(probs, unbiased=False)
        _mean = torch.mean(probs)
        return (_std / _mean) ** 2

    def compute_load_loss(self, logits, logits_noise, noise_std): # [B, Expert]
        assert logits_noise.dim()==2
        num_experts = logits_noise.shape[-1]
        threshold_per_item, _ = logits_noise.topk(self.k, dim=-1) # [B, K]
        threshold_per_item = threshold_per_item[:, -1] # [B]
        noise_required_to_win = threshold_per_item[:, None] - logits # [B, E]

        p = 1. - torch.distributions.Normal(0, noise_std).cdf(noise_required_to_win)
        p_mean = p.mean(0)

        return (torch.std(p_mean, unbiased=False) / torch.mean(p_mean)) ** 2

    def top_k_gating(self, x, skip_mask=None, sample_topk=0):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        logits_before = self.f_gate(x)
        if skip_mask:
            probs_before = torch.softmax(logits_before, dim=1) * skip_mask
        else:
            probs_before = torch.softmax(logits_before, dim=1)

        # add noise
        noise = torch.normal(torch.zeros_like(logits_before).float(), torch.ones_like(logits_before).float()/self.num_experts).cuda()
        # print('noise: ', noise.mean(), noise.min(), noise.max())
        # print('logits: ', logits.mean(), logits.min(), logits.max())
        logits = logits_before + noise

        if skip_mask:
            probs = torch.softmax(logits, dim=1) * skip_mask
        else:
            probs = torch.softmax(logits, dim=1)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6).detach()
        zeros = torch.zeros_like(probs, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        loss = 0
        
        loss = loss + self.compute_importance_loss(probs_before)
        loss = loss + self.compute_load_loss(logits_before, logits, 1./(self.num_experts * 1.))

        if self.concat:
            top_k_gates = top_k_gates.view(-1, 1)
            top_k_indices = top_k_indices.view(-1, 1)
            
            zeros = torch.zeros(top_k_gates.size(0), logits.size(1), device=x.device, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates, loss

class RobertaSelfAttentionMoASparse(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.head_size = config.attention_head_size
        self.att_experts = nn.ModuleList([
            nn.Linear(config.hidden_size, config.attention_head_size, bias=True)
            for _ in range(config.num_experts)
        ])
        self.query = MoE(config.hidden_size, self.att_experts, self.num_attention_heads, 
            dropout=None, concat=True, cvloss=config.cvloss, switchloss=config.switchloss, zloss=config.zloss)
        self.key = nn.Linear(config.hidden_size, config.attention_head_size)
        self.value = nn.Linear(config.hidden_size, config.attention_head_size)

        self.out_proj = nn.ModuleList([
            nn.Linear(config.attention_head_size, config.hidden_size, bias=True)
            for _ in range(config.num_experts)
        ])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder

        self.sample_topk = config.sample_topk
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        bsz, length, embed_dim = hidden_states.size()

        q, aux_loss = self.query(hidden_states, multiply_by_gates=False, sample_topk=self.sample_topk)

        # mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # key_layer = self.transpose_for_scores(self.key(hidden_states))
            # value_layer = self.transpose_for_scores(self.value(hidden_states))
            k = self.key(hidden_states)
            v = self.value(hidden_states)

        # query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = torch.einsum('bike,bje->bkij',q,k)
        assert list(attention_scores.size())==[bsz, self.num_attention_heads, length, length]

        attention_scores = attention_scores / math.sqrt(self.head_size)
        if attention_mask is not None:
            # print(f"The size of the attention mask is {attention_mask.size()}.")
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        # print(f"The size of the attention_probs is {attention_probs.size()}.")
        # print(f"The size of the value is {v.size()}.")
        context_layer = torch.einsum('bkij,bje->bike', attention_probs, v)

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.query.dispatch(context_layer, self.out_proj)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs, aux_loss
