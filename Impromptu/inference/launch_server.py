import io
import json
import base64
from typing import List, Dict, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Base64Bytes
from PIL import Image
import requests
import uvicorn
import argparse

from sglang.utils import launch_server_cmd, wait_for_server
import os


def start_qwen_server(
    qwen_ckpt_path,
    context_length: Optional[int] = None,
    tp_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    mem_fraction_static: Optional[float] = None,
    disable_cuda_graph: bool = False,
    max_running_requests: Optional[int] = None,
    max_prefill_tokens: Optional[int] = None,
    disable_radix_cache: bool = False,
):
    cmd_parts = [
        "/path/to/your/envs/sglang/bin/python -m sglang.launch_server",
        f"--model-path {qwen_ckpt_path}",
        "--chat-template=qwen2-vl",
    ]

    if context_length is not None:
        cmd_parts.append(f"--context-length {context_length}")
    if tp_size is not None:
        cmd_parts.append(f"--tp-size {tp_size}")
    if dp_size is not None:
        cmd_parts.append(f"--dp-size {dp_size}")
    if mem_fraction_static is not None:
        cmd_parts.append(f"--mem-fraction-static {mem_fraction_static}")
    if disable_cuda_graph:
        cmd_parts.append("--disable-cuda-graph")
    if max_running_requests is not None:
        cmd_parts.append(f"--max-running-requests {max_running_requests}")
    if max_prefill_tokens is not None:
        cmd_parts.append(f"--max-prefill-tokens {max_prefill_tokens}")
    if disable_radix_cache:
        cmd_parts.append("--disable-radix-cache")
    cmd = " \
                ".join(cmd_parts)

    vision_process, port = launch_server_cmd(
        f"""
        {cmd}
        """
    )

    # save port
    port_file_dir = os.path.dirname(args.port_file)
    if port_file_dir and not os.path.exists(port_file_dir):
        os.makedirs(port_file_dir, exist_ok=True)
        print(f"✅ Created directory: {port_file_dir}")

    with open(f"{args.port_file}", "w") as f:
        f.write(str(port))

    wait_for_server(f"http://localhost:{port}")
    print(f"✅ Qwen service ready at port {port}")
    return vision_process, port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Qwen server with optional checkpoint path."
    )
    parser.add_argument(
        "--qwen_ckpt_path", type=str, required=True, help="Path to Qwen checkpoint."
    )
    parser.add_argument(
        "--port_file", type=str, required=True, help="Path to port file."
    )
    parser.add_argument(
        "--context-length", type=int, dest="context_length", help="Model context length."
    )
    parser.add_argument(
        "--tp-size", type=int, dest="tp_size", help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size", type=int, dest="dp_size", help="Data parallel size."
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        dest="mem_fraction_static",
        help="Static memory fraction.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph.",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_requests",
        help="Maximum running requests.",
    )
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        dest="max_prefill_tokens",
        help="Maximum prefill tokens.",
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Disable radix cache.",
    )
    args = parser.parse_args()

    _, QWEN_PORT = start_qwen_server(
        args.qwen_ckpt_path,
        context_length=args.context_length,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=args.disable_cuda_graph,
        max_running_requests=args.max_running_requests,
        max_prefill_tokens=args.max_prefill_tokens,
        disable_radix_cache=args.disable_radix_cache,
    )
