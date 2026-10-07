# Serving a 235B model with vLLM-TPU across a multi-host slice

A 235B model in bfloat16 doesn't fit on one 8-chip host. We served **Qwen3-235B-A22B** with
vLLM-TPU on a **multi-host Trillium (v6e) TPU VM slice**, with 16-way tensor parallelism across hosts over Ray. The
weights were read from Cloud Storage through gcsfuse with a RAM-backed file cache, so no host needed
local disk for the full model.

> Run Feb 2026, on the public base model. Placeholders: `TPU_NAME`, `ZONE`,
> `BUCKET`, `MODEL_DIR` (a prefix in the bucket holding HF safetensors), `HEAD_IP` (internal IP of
> worker 0).

Install vLLM-TPU on every worker first ([01](01-airgapped-install.md), step 5a).

## 1. Start Ray

On worker 0:

```bash
python -m ray.scripts.scripts start --head --port=6379
```

On every other worker:

```bash
python -m ray.scripts.scripts start --address="HEAD_IP:6379"
```

## 2. Mount the weights with a RAM-backed cache, on all workers

```bash
W="gcloud alpha compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=all --command"

$W "sudo mkdir -p /mnt/gcs-cache /mnt/model"
$W "sudo mount -t tmpfs -o size=200G tmpfs /mnt/gcs-cache"
$W "sudo chown -R \$USER:\$USER /mnt/gcs-cache /mnt/model"

$W "gcsfuse --profile=aiml-serving --client-protocol=grpc \
      --cache-dir=/mnt/gcs-cache --file-cache-max-size-mb=204800 \
      --file-cache-parallel-downloads-per-file=32 \
      --file-cache-max-parallel-downloads=256 \
      --file-cache-download-chunk-size-mb=128 \
      --metadata-cache-ttl-secs=-1 --kernel-list-cache-ttl-secs=-1 \
      --only-dir MODEL_DIR/ -o ro BUCKET /mnt/model"
```

`--profile=aiml-serving` applies gcsfuse's preset for read-heavy model loading. The parallel-download
flags matter most: without them, the first load of hundreds of GB through FUSE is serialized.

## 3. Load and generate, on worker 0

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"
os.environ["VLLM_HOST_IP"] = "HEAD_IP"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from vllm import LLM, SamplingParams
llm = LLM(
    model="/mnt/model",
    tensor_parallel_size=16,              # total chips across all hosts
    distributed_executor_backend="ray",
    max_model_len=8192,
    gpu_memory_utilization=0.95,          # the name says GPU; it governs TPU HBM too
    max_num_batched_tokens=512,
    max_num_seqs=20,
)
print(llm.generate("How does a rainbow work?", SamplingParams())[0].outputs[0].text)
```

**Memory budget.** Qwen3-235B in bfloat16 is ~470 GB of weights. At 32 GB of HBM per Trillium
chip, 16-way tensor parallelism gives 512 GB. At `gpu_memory_utilization=0.95` that leaves only
~15 GB for KV cache. Hence the small `max_num_batched_tokens` and `max_num_seqs`. For
real throughput, use a larger slice (and a matching `tensor_parallel_size`) or an FP8 checkpoint.
