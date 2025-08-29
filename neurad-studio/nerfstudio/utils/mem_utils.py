import torch

def log_mem(tag=""):
    """打印当前显存占用情况"""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    res   = torch.cuda.memory_reserved() / 1024**2
    peak  = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[MEM] {tag:30s} alloc={alloc:.1f}MB reserved={res:.1f}MB peak={peak:.1f}MB", flush=True)
