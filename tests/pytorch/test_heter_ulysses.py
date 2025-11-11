# torchrun --nproc_per_node 4 tests/pytorch/test_heter_ulysses.py
import os
import torch
import torch.distributed as dist
from transformer_engine.pytorch import DotProductAttention


def test_heter_ulysses():
    headnum_tot = 16
    seqlen_tot = 4096
    bsz = 1
    kv_channels = 128
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend='nccl')
    cp_stream = torch.cuda.Stream()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_args = {
        "num_attention_heads": headnum_tot, 
        "kv_channels": kv_channels, 
        "num_gqa_groups": 8, 
        "qkv_format": "bshd",
        "cp_group": dist.group.WORLD,
        "cp_stream": cp_stream,
        "cp_comm_type": "a2a",
    }
    attn_homo = DotProductAttention(**init_args)
    init_args.update({
        "seqlen_tot": 4096, 
        "seqlen_per_gpu": torch.Tensor([seqlen_tot//world_size] * world_size),
        "headnum_per_gpu": torch.Tensor([headnum_tot//world_size] * world_size),
    })
    attn_heter = DotProductAttention(**init_args)

    q = torch.rand(bsz, seqlen_tot//world_size, headnum_tot, kv_channels, device='cuda')
    k = torch.rand(bsz, seqlen_tot//world_size, headnum_tot, kv_channels, device='cuda')
    v = torch.rand(bsz, seqlen_tot//world_size, headnum_tot, kv_channels, device='cuda')
    homo_out = attn_homo(q,k,v)
    heter_out = attn_heter(q,k,v)
    assert torch.equal(homo_out, heter_out)
    print("heterogeneous ulysses test passed!!!")


if __name__ == '__main__':
    test_heter_ulysses()