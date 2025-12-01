# torchrun --nproc_per_node 2 tests/pytorch/test_heter_ulysses.py
import os
import copy
import torch
import torch.distributed as dist
from transformer_engine.pytorch import DotProductAttention


def get_batch_on_this_cp_rank(batch, local_rank, cp_size, seqlen_per_gpu, homo):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    if cp_size > 1:
        for idx in range(len(batch)):
            val = batch[idx]
            if homo:
                val = val.view(
                    val.shape[0],
                    2 * cp_size,
                    val.shape[1] // (2 * cp_size),
                    *val.shape[2 :],
                )
                index = torch.zeros(2, dtype=torch.int64, device=val.device)
                index[0].fill_(local_rank)
                index[1].fill_(2 * cp_size - local_rank - 1)
                val = val.index_select(1, index)
                val = val.view(val.shape[0], -1, *val.shape[3 :])
                batch[idx] = val
            else:
                # heterogeneous ulysses, only support pure ulysses, no need for load balance now
                seqlen_splits = seqlen_per_gpu.tolist()
                val_splits = torch.split(val, seqlen_splits, dim=1)
                val = val_splits[local_rank]
                batch[idx] = val

    return batch


def reorder_seq_chunks(x, cp_size):
    chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=x.device)
    for rank in range(cp_size):
        chunk_ids[rank] = 2 * rank
        chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
    x = x.view(x.shape[0], 2 * cp_size, x.shape[1] // (2 * cp_size), *x.shape[2:])
    x = x.index_select(dim=1, index=chunk_ids)
    x = x.view(x.shape[0], -1, *x.shape[3:])
    return x


class PadAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, dim=0):
        """
        input_tensor: local tensor to gather (shape: [L_local, ...] along dim)
        returns: concatenated tensor from all ranks (unpadded)
        """
        world_size = dist.get_world_size()
        device = input_tensor.device
        # gather lengths
        local_len = input_tensor.shape[dim]
        local_len_t = torch.tensor([local_len], device=device, dtype=torch.long)
        all_lens = [torch.zeros_like(local_len_t) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len_t)
        all_lens = [int(x.item()) for x in all_lens]
        max_len = max(all_lens)
        pad_len = max_len - local_len

        # pad local
        if pad_len > 0:
            pad_shape = list(input_tensor.shape)
            pad_shape[dim] = pad_len
            pad_tensor = torch.zeros(*pad_shape, dtype=input_tensor.dtype, device=device)
            input_padded = torch.cat([input_tensor, pad_tensor], dim=dim)
        else:
            input_padded = input_tensor

        # allocate per-rank buffers and all_gather into them
        # Use empty_like so dtype/device match
        out_list = [torch.empty_like(input_padded) for _ in range(world_size)]
        dist.all_gather(out_list, input_padded)

        # stack -> tensor shape (world_size, *input_padded.shape)
        stacked = torch.stack(out_list, dim=0)

        # simpler: iterate indices
        pieces = [stacked[i].narrow(dim, 0, l) for i, l in enumerate(all_lens)]
        gathered = torch.cat(pieces, dim=dim)

        # save for backward
        ctx.dim = dim
        ctx.device = device
        ctx.all_lens = all_lens
        ctx.max_len = max_len
        ctx.input_padded_shape = tuple(input_padded.shape)
        # we don't need to save input tensors themselves
        return gathered

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: gradient w.r.t the gathered (concatenated) output
        We must produce gradient matching input_tensor (unpadded local shape).
        Strategy:
         - split grad_output into per-rank pieces according to ctx.all_lens
         - pad each piece to ctx.max_len along ctx.dim -> form local send_list
         - call dist.reduce_scatter(recv, send_list) to get summed gradients destined for this rank
         - crop recv to local_len and return as grad for input_tensor
        """
        dim = ctx.dim
        device = ctx.device
        all_lens = ctx.all_lens
        max_len = ctx.max_len

        # debug prints (optional)
        # print(f"[PadAllGather.backward] grad_output.shape={tuple(grad_output.shape)} on {device}")

        # split grad_output into per-rank pieces
        pieces = []
        start = 0
        for l in all_lens:
            if l == 0:
                # edge case
                pieces.append(None)
            else:
                pieces.append(grad_output.narrow(dim, start, l).contiguous())
            start += l

        # for each piece, pad to max_len along dim to form send list
        send_list = []
        for p in pieces:
            if p is None:
                # create zeros of padded shape
                pad_shape = list(grad_output.shape)
                pad_shape[dim] = max_len
                send_list.append(torch.zeros(*pad_shape, dtype=grad_output.dtype, device=device))
            else:
                cur_len = p.shape[dim]
                pad_len = max_len - cur_len
                if pad_len > 0:
                    pad_shape = list(p.shape)
                    pad_shape[dim] = pad_len
                    pad_tensor = torch.zeros(*pad_shape, dtype=p.dtype, device=device)
                    p_padded = torch.cat([p, pad_tensor], dim=dim)
                else:
                    p_padded = p
                send_list.append(p_padded)

        # prepare recv buffer for this rank (shape == input_padded_shape)
        recv = torch.zeros(ctx.input_padded_shape, dtype=grad_output.dtype, device=device)

        # dist.reduce_scatter will take a list of tensors (one per rank) from each rank,
        # sum them elementwise across ranks, and scatter the results so that recv gets the i-th sum.
        # We need send_list to be of length world_size.
        dist.reduce_scatter(recv, send_list)

        # Now recv contains the summed gradients destined for this rank's padded input.
        # Crop to local original length (first element of all_lens for this rank)
        # But we need to know *which* rank we are (get_rank)
        rank = dist.get_rank()
        local_len = all_lens[rank]
        if local_len == recv.shape[dim]:
            grad_input = recv
        else:
            grad_input = recv.narrow(dim, 0, local_len).contiguous()

        # Return gradient for input_tensor and None for 'dim' argument
        return grad_input, None


def pad_all_gather_with_autograd(x: torch.Tensor, dim=0):
    return PadAllGather.apply(x, dim)


def test_heter_ulysses():
    torch.autograd.set_detect_anomaly(True)
    num_attention_heads = 2
    num_gqa_groups = 2
    seqlen_tot = 2048
    bsz = 1
    kv_channels = 8
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend='nccl')
    cp_stream = torch.cuda.Stream()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_args = {
        "num_attention_heads": num_attention_heads, 
        "kv_channels": kv_channels, 
        "num_gqa_groups": num_gqa_groups, 
        "qkv_format": "bshd",
        "cp_group": dist.group.WORLD,
        "cp_stream": cp_stream,
        "cp_comm_type": "a2a",
    }
    attn_homo = DotProductAttention(**init_args)
    # seqlen_per_gpu = torch.tensor([seqlen_tot//world_size] * world_size, dtype=torch.long)
    seqlen_per_gpu = torch.tensor([1414, 634], dtype=torch.long)
    # headnum_per_gpu_kv = torch.tensor([num_gqa_groups//world_size] * world_size, dtype=torch.long)
    headnum_per_gpu_kv = torch.tensor([1, 1], dtype=torch.long)
    init_args.update({
        "seqlen_tot": seqlen_tot, 
        "seqlen_per_gpu": seqlen_per_gpu,
        "headnum_per_gpu_kv": headnum_per_gpu_kv,
    })
    attn_heter = DotProductAttention(**init_args)

    if local_rank == 0:
        q = torch.rand(bsz, seqlen_tot, num_attention_heads, kv_channels, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        k = torch.rand(bsz, seqlen_tot, num_gqa_groups, kv_channels, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        v = torch.rand(bsz, seqlen_tot, num_gqa_groups, kv_channels, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    else:
        q = torch.empty(bsz, seqlen_tot, num_attention_heads, kv_channels, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        k = torch.empty(bsz, seqlen_tot, num_gqa_groups, kv_channels, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        v = torch.empty(bsz, seqlen_tot, num_gqa_groups, kv_channels, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    q_homo, k_homo, v_homo = get_batch_on_this_cp_rank(
        [copy.deepcopy(q), copy.deepcopy(k), copy.deepcopy(v)],
        local_rank, world_size, None, True,
    )
    q_heter, k_heter, v_heter = get_batch_on_this_cp_rank(
        [q, k, v],
        local_rank, world_size, seqlen_per_gpu, False,
    )
    q_homo = q_homo.contiguous().clone()
    k_homo = k_homo.contiguous().clone()
    v_homo = v_homo.contiguous().clone()
    q_heter = q_heter.contiguous().clone()
    k_heter = k_heter.contiguous().clone()
    v_heter = v_heter.contiguous().clone()

    q_homo.retain_grad()
    k_homo.retain_grad()
    v_homo.retain_grad()
    q_heter.retain_grad()
    k_heter.retain_grad()
    v_heter.retain_grad()

    homo_out = attn_homo(q_homo, k_homo, v_homo)
    heter_out = attn_heter(q_heter, k_heter, v_heter)

    # all_gather_object will break computing graph, so use all_gather + padding + unpadding
    homo_out = pad_all_gather_with_autograd(homo_out, dim=1) 
    homo_out = reorder_seq_chunks(homo_out, world_size)

    heter_out = pad_all_gather_with_autograd(heter_out, dim=1)

    assert torch.equal(homo_out, heter_out)
    print("heterogeneous ulysses forward test passed!!!")
    
    # backward
    loss_homo = (homo_out * homo_out).mean()
    loss_heter = (heter_out * heter_out).mean()

    loss_homo.backward()
    loss_heter.backward()

    q_homo_grad = reorder_seq_chunks(pad_all_gather_with_autograd(q_homo.grad, dim=1), world_size)
    k_homo_grad = reorder_seq_chunks(pad_all_gather_with_autograd(k_homo.grad, dim=1), world_size)
    v_homo_grad = reorder_seq_chunks(pad_all_gather_with_autograd(v_homo.grad, dim=1), world_size)

    q_heter_grad = pad_all_gather_with_autograd(q_heter.grad, dim=1)
    k_heter_grad = pad_all_gather_with_autograd(k_heter.grad, dim=1)
    v_heter_grad = pad_all_gather_with_autograd(v_heter.grad, dim=1)

    #(lj) NOTE: due to the non-deterministic of flash attn backward, we can't use torch.equal() for bwd result comparison.
    q_diff = torch.max(torch.abs(q_homo_grad - q_heter_grad) / (torch.abs(q_heter_grad) + 1e-13))
    k_diff = torch.max(torch.abs(k_homo_grad - k_heter_grad) / (torch.abs(k_heter_grad) + 1e-13))
    v_diff = torch.max(torch.abs(v_homo_grad - v_heter_grad) / (torch.abs(v_heter_grad) + 1e-13))

    assert q_diff.item() < 1e-2
    assert k_diff.item() < 1e-2
    assert v_diff.item() < 1e-2
    print("heterogeneous ulysses backward test passed!!!")


if __name__ == '__main__':
    test_heter_ulysses()