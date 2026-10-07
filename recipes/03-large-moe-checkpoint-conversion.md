# Converting a 235B MoE MaxText checkpoint to Hugging Face format

After continued pretraining of **Qwen3-235B-A22B** in MaxText, we needed Hugging Face safetensors
so we could serve and evaluate the model with vLLM. The conversion runs on CPU and is bounded by host
memory. A standard TPU VM host doesn't have enough
([MaxText #3418](https://github.com/AI-Hypercomputer/maxtext/issues/3418) reports running out of
memory on a ~400 GB v5p host).

We ran it as a one-off Kubernetes Job on an **ultramem CPU node** in the same GKE cluster as the
training. The node pool scales from zero, uses Spot capacity, and costs nothing when idle.

> Conversion by Dr. Faith Mutinda (Nov 2025, MaxText `to_huggingface.py` of that era). The resource
> numbers below are **what worked, not a measured minimum**. We didn't record peak memory.

Placeholders: `CLUSTER`, `REGION`, `ZONE`, `PROJECT_ID`, `GSA` (node service account), `KSA`
(Kubernetes service account with read/write access to your buckets), `BUCKET`, `IMAGE` (a MaxText
image, see [01](01-airgapped-install.md)).

## 1. A scale-to-zero ultramem node pool

```bash
gcloud container node-pools create high-mem-cpu-pool \
  --cluster=CLUSTER --region=REGION --node-locations=ZONE \
  --machine-type=m4-ultramem-112 \
  --spot \
  --num-nodes=1 --enable-autoscaling --min-nodes=0 --max-nodes=1 \
  --service-account=GSA
```

`m4-ultramem-112` has ~2.9 TiB of memory. The Job below requests most of it.

## 2. The conversion Job

Two choices matter:

- **`JAX_PLATFORMS=cpu` and `skip_jax_distributed_system=True`.** The converter is a single CPU
  process, so don't let JAX look for accelerators or peers.
- **A memory-backed `emptyDir` at `/dev/shm`** for staging files locally. Container `/dev/shm`
  defaults to 64 MB.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ckpt-to-hf
  namespace: maxtext
spec:
  backoffLimit: 0
  template:
    spec:
      serviceAccountName: KSA
      restartPolicy: Never
      containers:
      - name: convert
        image: IMAGE
        env:
        - name: JAX_PLATFORMS
          value: "cpu"
        - name: HF_HUB_OFFLINE          # air-gapped: fail fast instead of timing out
          value: "1"
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euxo pipefail
          python src/MaxText/utils/ckpt_conversion/to_huggingface.py \
            src/MaxText/configs/base.yml \
            model_name=qwen3-235b-a22b \
            skip_jax_distributed_system=True \
            checkpoint_storage_concurrent_gb=350 \
            load_parameters_path=gs://BUCKET/RUN/checkpoints/STEP/items \
            base_output_directory=gs://BUCKET/RUN-STEP-HF
        resources:
          requests: {memory: "2800Gi", cpu: "100"}
          limits:   {memory: "2800Gi", cpu: "100"}
        volumeMounts:
        - {name: dshm, mountPath: /dev/shm}
      volumes:
      - name: dshm
        emptyDir: {medium: Memory, sizeLimit: 2000Gi}
```

`checkpoint_storage_concurrent_gb` caps how much checkpoint data Orbax reads at once. Its default
(96 in current MaxText) is tuned for TPU hosts. Raising it on a large-memory node shortens the read.

The script path has since moved, to `src/maxtext/checkpoint_conversion/to_huggingface.py` in 2026
releases.

## 3. Without Hugging Face Hub access

`to_huggingface.py` takes the model architecture from a table built into MaxText, but it **loads the
tokenizer from the Hub by model ID** (`AutoTokenizer.from_pretrained(HF_IDS[model_name])`). On an
air-gapped cluster that call fails.

**MaxText 0.2.5 and later:** stage the base model's tokenizer files in a local directory and pass
`--hf_model_path=/path/to/tokenizer_dir`. The flag accepts a local path.

**Older versions** (ours was Oct 2025) have no such flag. Two workarounds:

1. **Pre-populate the Hugging Face cache.** On an internet-connected machine, run
   `huggingface-cli download Qwen/Qwen3-235B-A22B-Thinking-2507 --include "tokenizer*" "*.json" --cache-dir hf_cache`
   (the ID must match MaxText's `HF_IDS` entry for your `model_name` exactly),
   stage `hf_cache` in a bucket, copy it into the container, and set `HF_HOME` to it together with
   `HF_HUB_OFFLINE=1`. `from_pretrained` will then resolve the ID from the cache.
2. **Patch the call** so it points at a local directory holding the base model's tokenizer files.

The converter version we used (Oct 2025) made the same Hub call. We don't have a record of which
workaround our container used.

## 4. What came out

- Input: one ~16 TiB Orbax checkpoint (parameters plus optimizer state). The converter only reads
  parameters.
- Output: **294 safetensors shards, 940 GB (~875 GiB)**. That is ~4 bytes per parameter, i.e.
  **float32**, because MaxText's `weight_dtype` defaults to `float32`.
  ([#3489](https://github.com/AI-Hypercomputer/maxtext/issues/3489) asks for half-precision export.)
  Current MaxText documentation describes a `weight_dtype=bfloat16` option, which should halve the
  output. We haven't tested it on this model.

For serving a model this size with vLLM-TPU across hosts, see [04](04-multihost-vllm-inference.md).
