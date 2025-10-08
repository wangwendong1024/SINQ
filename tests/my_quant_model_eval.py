"""
written by SINQ authors
"""
import os
import argparse
from timeit import default_timer as timer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from torch.utils.data import DataLoader

import os
from functools import partial
from tqdm import tqdm
import numpy as np

from eval_my.evaluate_ import evaluate_model

import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# from patch_model_demo import quantize_model
use_sinq = True
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def get_model(model_name, nbits=None, group_size=128, tiling_mode='1D', method='dual', axis=1, device='cpu'):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if nbits is None:
        return model, tokenizer

    quant_config = BaseQuantizeConfig(
        nbits=nbits, group_size=group_size, axis=axis, tiling_mode=tiling_mode,
        method=method
    )

    start_time = timer()
    AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=device
    )
    end_time = timer()
    duration = (end_time - start_time)
    print(f"Time to quantize model: {duration:.2f} seconds")

    print(
        "CUDA memory allocated and reserved (GB):",
        torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9  # in GB
    )
    memory_alloc = torch.cuda.memory_allocated() / 1e9
    print(model)
    return model, tokenizer, memory_alloc
def main():
    parser = argparse.ArgumentParser(description="Quantized model evaluation")

    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name, e.g., Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, etc."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0",
        help="Device to use: 'cuda:0' or 'cpu'"
    )
    parser.add_argument("--nbits", type=int, default=4, help="[3,4,5,8]")
    parser.add_argument("--group_size", type=int, default=64, help="[32,64,128,256]")
    parser.add_argument("--axis", type=int, default=1, help="Fixed to 1")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--dataset_name", type=str, default="wikitext2", help="Dataset name")
    parser.add_argument("--tiling_mode", type=str, default="1D", help="Can be 1D or 2D")
    parser.add_argument(
        "--method", type=str,
        default="sinq",
        help="Quantization method: sinq, noz, hqq, quantAux, awq, l1"
    )

    args = parser.parse_args()

    # 自动检测 CUDA 是否可用
    if "cuda" in args.device and not torch.cuda.is_available():
        print("⚠️ CUDA not available, switching to CPU.")
        args.device = "cpu"

    print(f"🔥 Using device: {args.device}")

    # 加载模型
    model, tokenizer, memory = get_model(
        args.model_name,
        nbits=args.nbits,
        group_size=args.group_size,
        axis=args.axis,
        device=args.device,
        tiling_mode=args.tiling_mode,
        method=args.method
    )

    # 编译模型（仅在 CUDA 下生效）
    if args.device.startswith("cuda"):
        model = torch.compile(model)
    print(f"Allocated memory (GB): {memory:.2f}")

    # 评估模型
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks="",
        eval_ppl=args.dataset_name,
        batch_size=args.batch_size
    )

    # 取出结果
    task_results = results.get(args.dataset_name, None)
    print(f"✅ Evaluation done for {args.model_name}")
    print(f"nbits={args.nbits}, method={args.method}, result={task_results}, memory={memory:.2f} GB")

if __name__ == '__main__':

    main()


