import sys
import json
import multiprocessing as mp
from tqdm import tqdm
from chonky.markup_remover import MarkupRemover
from chonky import ParagraphSplitter
import os
import torch

PARALLEL_PROCESSES = 50

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Globals inside each worker
torch_remover = None
torch_splitter = None


def init_worker(model_id, devices):
    """
    Initialize MarkupRemover and ParagraphSplitter in each worker.
    Each worker picks its device based on its unique worker id.
    """
    global torch_remover, torch_splitter
    torch_remover = MarkupRemover()
    # Determine worker index (1-based) -> zero-based
    worker_info = mp.current_process()._identity
    if worker_info:
        idx = worker_info[0] - 1
    else:
        idx = 0
    device = devices[idx % len(devices)]
    torch_splitter = ParagraphSplitter(model_id=model_id, device=device)


def process_item(item):
    """
    Remove markup and split a single record's contents.
    Returns a list of chunked record dicts.
    """
    text = item.get("contents", "")
    if not text:
        return []
    plain = torch_remover(text)
    chunks = list(torch_splitter(plain))
    out = []
    for chunk in chunks:
        rec = item.copy()
        rec["contents"] = chunk
        out.append(rec)
    return out


def main(
    input_path,
    output_path,
    model_id="mirth/chonky_distilbert_base_uncased_1",
    base_device="cuda",
    num_procs=None,
):
    """
    Spawn a pool of workers (using 'spawn') each loading its own Chonky models.
    Caps processes to 20 per GPU to avoid OOM. Automatically rounds to nearest lower.
    Each worker is assigned a device in round-robin order based on its worker id.

    Usage:
        python optimized_chunker_mp.py in.json out.jsonl [model_id] [base_device] [num_procs]
    """
    # Determine GPUs and process cap
    gpu_count = torch.cuda.device_count() or 1
    max_procs = gpu_count * PARALLEL_PROCESSES
    desired = num_procs or mp.cpu_count()
    total_procs = min(int(desired), max_procs)

    # Build device list round-robin
    devices = [f"{base_device}:{i}" for i in range(gpu_count)]

    # Use spawn to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=total_procs, initializer=init_worker, initargs=(model_id, devices)
    )

    is_jsonl = input_path.lower().endswith(".jsonl")
    chunk_id = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        if is_jsonl:
            source_iter = (
                json.loads(line) for line in open(input_path, "r", encoding="utf-8")
            )
        else:
            data = json.load(open(input_path, "r", encoding="utf-8"))
            source_iter = iter(data)

        for out_chunks in tqdm(
            pool.imap_unordered(process_item, source_iter), desc="Chunking", unit="rec"
        ):
            for rec in out_chunks:
                rec["chunk_id"] = chunk_id
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunk_id += 1

    pool.close()
    pool.join()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python optimized_chunker_mp.py input.json[.jsonl] output.jsonl [model_id] [base_device] [num_procs]"
        )
        sys.exit(1)
    args = sys.argv[1:]
    # parse num_procs if provided
    if len(args) >= 5:
        args[4] = int(args[4])
    main(*args)
