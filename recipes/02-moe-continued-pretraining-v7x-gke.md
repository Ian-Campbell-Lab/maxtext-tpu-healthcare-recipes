# Continued pretraining of Qwen3-235B-A22B on 64 Ironwood (v7x) chips with MaxText on GKE

We ran continued pretraining of **Qwen3-235B-A22B** (128-expert MoE, 22B active parameters) on
**one 4x4x4 Ironwood slice: 64 v7x chips across 16 hosts**, scheduled as a plain Kubernetes Indexed
Job on GKE. We didn't use XPK or Pathways for this run. It ran for **11 days without a gap in
logging**.

> Run Nov 2025. The config values below are read back from the run's own
> TensorBoard text summaries, not reconstructed from memory. Placeholders: `PROJECT_ID`, `CLUSTER`,
> `REGION`, `ZONE`, `GSA`, `KSA`, `RESERVATION`, `BUCKET`, `IMAGE`.

## Results

| | |
|---|---|
| Hardware | 1 slice, v7x 4x4x4 = 64 chips / 128 JAX devices / 16 hosts |
| Global batch | 768 sequences × 8,192 tokens = 6.29M tokens/step (~99.6% non-padding) |
| Step time | 42.65 s median (first step ~114 s including compilation) |
| Throughput | ~147.5k tokens/s; 192.7 TFLOP/s per device as reported by MaxText |
| Trained | 22,800 steps ≈ **143B tokens** in 11 days 6.7 h of continuous logging |
| Loss | 1.67 → 0.42 (continued pretraining on a domain corpus; not comparable to public pretraining loss) |

## 1. Cluster and node pool

A workload policy keeps the 16 hosts in one physical 4x4x4 topology:

```bash
gcloud compute resource-policies create workload-policy tpu7x-workload-policy \
  --type HIGH_THROUGHPUT --accelerator-topology 4x4x4 \
  --project PROJECT_ID --region REGION

gcloud container node-pools create ironwood-multi \
  --cluster=CLUSTER --region=REGION --node-locations=ZONE \
  --machine-type=tpu7x-standard-4t --num-nodes=16 \
  --placement-policy=tpu7x-workload-policy \
  --reservation=RESERVATION --reservation-affinity=specific \
  --service-account=GSA \
  --node-labels=cloud.google.com/gke-tpu-accelerator=tpu7x,cloud.google.com/gke-tpu-topology=4x4x4 \
  --workload-metadata=GKE_METADATA
```

`GKE_METADATA` enables
[Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity).
The cluster must have it enabled (`--workload-pool=PROJECT_ID.svc.id.goog`). Bind the Job's
Kubernetes service account `KSA` to a Google service account so that only this Job gets bucket
access. Avoid `GCE_METADATA`, which hands the node service account's credentials to
every Pod on the node.

Also add a small CPU system pool (e.g. `n1-standard-8`, autoscaling 1–3) so cluster services don't
compete for the TPU nodes.

## 2. One Indexed Job, one Pod per host

GKE injects the TPU multi-host environment (`TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`) into each Pod of
an Indexed Job that runs behind a headless Service, so JAX's distributed initialization needs no
coordinator configuration.

```yaml
apiVersion: v1
kind: Service
metadata: {name: tpu-headless, namespace: maxtext}
spec:
  clusterIP: None
  selector: {job-name: qwen3-235b-cpt}
---
apiVersion: batch/v1
kind: Job
metadata: {name: qwen3-235b-cpt, namespace: maxtext}
spec:
  completions: 16            # 16 hosts x 4 chips = one 4x4x4 slice
  parallelism: 16
  completionMode: Indexed
  backoffLimit: 0
  template:
    spec:
      subdomain: tpu-headless
      serviceAccountName: KSA
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-nodepool: ironwood-multi
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 4x4x4
      tolerations:
      - {key: google.com/tpu, operator: Equal, value: present, effect: NoSchedule}
      containers:
      - name: maxtext
        image: IMAGE
        resources:
          requests: {google.com/tpu: "4"}
          limits:   {google.com/tpu: "4"}
        env:
        - name: LIBTPU_INIT_ARGS
          value: "SEE SECTION 4"
        command: ["/bin/bash", "-lc"]
        args: ["python3 -m MaxText.train src/MaxText/configs/base.yml SEE_SECTION_3"]
```

Before committing the slice to a long run, do a 20-step smoke test with `model_name=llama3.1-8b
dataset_type=synthetic steps=20 enable_checkpointing=false`. It checks scheduling, multi-host
initialization, and bucket permissions in a few minutes.

## 3. MaxText configuration

Start from a parameters-only MaxText checkpoint converted from the public Hugging Face weights
(MaxText's `to_maxtext` converter, with `scan_layers=true`). Load it with `load_parameters_path`. To
resume after an interruption, restore full state (parameters plus optimizer state) instead.

```text
model_name=qwen3-235b-a22b
run_name=RUN  base_output_directory=gs://BUCKET/
load_parameters_path=gs://BUCKET/qwen3-235b-scanned/0/items

# Parallelism: pure FSDP across all 128 devices; every other axis = 1.
ici_fsdp_parallelism=-1  ici_expert_parallelism=1  dcn_data_parallelism=-1

per_device_batch_size=6  max_target_length=8192  gradient_accumulation_steps=1

opt_type=adamw  adam_b1=0.9  adam_b2=0.95  adam_eps=1e-8  adam_weight_decay=0.1
learning_rate=3e-5  warmup_steps_fraction=0.05  cosine_learning_rate_final_fraction=1.0
steps=38530

dtype=bfloat16  grad_dtype=bfloat16  weight_dtype=float32

remat_policy=custom
  query_proj=remat key_proj=remat value_proj=remat out_proj=remat context=remat
  mlpwi=remat mlpwi_0=remat mlpwi_1=remat mlpwo=remat
  decoder_layer_input=offload

sparse_matmul=true  megablox=true  capacity_factor=-1  use_custom_sort_vjp=true
tile_batch_seq=512  tile_embed_dim=1024  tile_mlp_dim=2048
load_balance_loss_weight=0.001

attention=autoselected  sa_block_q=1024  sa_block_kv=1024  sa_use_fused_bwd_kernel=true
scan_layers=true  shardy=true

dataset_type=grain  packing=false  grain_worker_count=2
checkpoint_period=200  async_checkpointing=true
```

Choices worth noting:

- **Pure FSDP won.** We tried many combinations of tensor, pipeline, and sequence parallelism, and
  none beat FSDP alone over all 128 devices on this single slice. The 128-expert model ran with
  dropless MoE (`capacity_factor=-1`) through megablox grouped matmuls and no expert parallelism.
  If you have one slice and a fast ICI torus, start with pure FSDP and make other layouts prove
  themselves against it.
- **Remat plus host offload instead of a smaller batch.** Every projection and the attention
  context are recomputed in the backward pass, and decoder-layer inputs are offloaded to host
  memory. With that setup the run fit 6 sequences of 8K per device alongside float32 master
  weights. We didn't search for the least aggressive policy that also fits.
- **`cosine_learning_rate_final_fraction=1.0`** turns the cosine schedule into warmup followed by a
  constant rate. That's a reasonable default for continued pretraining when you may stop early: the
  run ended at step ~22,800 of a 38,530-step schedule without leaving the learning rate partway
  through a decay.
- **`packing=false`** because the data was packed offline (next section).

## 3a. Training data: pre-tokenized, pre-packed ArrayRecord

Tokenizing and packing inside the training job wastes host CPU on every run. More importantly, it
gives you no control over document boundaries. We do both offline, once:

1. **Tokenize** each document, then **bin-pack** documents into 8,192-token windows
   (best-fit-decreasing). Each document in a window gets its own **segment ID** (1, 2, 3, …;
   0 = padding), and its **positions restart at 0**, so attention stays within each document and
   position encodings are per document.
2. **Write one `tf.train.Example` per window** with five int64 features, 5,120 windows per
   ArrayRecord shard. `group_size:1` keeps every record individually addressable, which Grain needs
   for random access and global shuffling.

```python
from array_record.python.array_record_module import ArrayRecordWriter
import tensorflow as tf

def feat(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

writer = ArrayRecordWriter("shard-00000.arrayrecord", "group_size:1")
for w in windows:   # each a dict of five length-8192 int arrays
    ex = tf.train.Example(features=tf.train.Features(feature={
        k: feat(w[k]) for k in ("inputs", "inputs_segmentation", "inputs_position",
                                "targets_segmentation", "targets_position")}))
    writer.write(ex.SerializeToString())
writer.close()
```

**Loss masking comes free with this format.** A token whose `targets_segmentation` is 0 is
excluded from the loss, so tokens that should condition the model but never be trained on just
get their target segment set to 0 at packing time.

MaxText settings for this format:

```text
dataset_type=grain  grain_file_type=arrayrecord  grain_train_files=/path/to/shards/*
tokenize_train_data=false  packing=false  add_bos=false  add_eos=false
train_data_columns=['inputs']
```

**One gotcha, and our own lesson.** In late-2025 MaxText the Grain ArrayRecord path discards
precomputed segmentation in **two** places:

1. `ParseFeatures` parses only `train_data_columns`, so the extra features never get read.
2. The pad step (`PadOrTrimToMaxLength`) **unconditionally** rewrites `<col>_segmentation` as
   "token is not padding" and `<col>_position` as `0…max_length-1`, even when the arrays already
   exist.

The result is that every document in a window becomes one segment: attention crosses documents,
positions run continuously, and your loss masks are gone. If your document separator is also the
tokenizer's pad ID (Qwen3 uses `<|endoftext|>` for both), every separator is also silently
excluded from the loss.

The fix has three parts, all in `input_pipeline/_input_pipeline_utils.py`:

```python
COLS = ("inputs", "inputs_segmentation", "inputs_position",
        "targets_segmentation", "targets_position")

# 1. ParseFeatures: parse all five columns, not just train_data_columns
parsed = tf.io.parse_example(
    example, {c: tf.io.FixedLenSequenceFeature([], dtype=self.dtype, allow_missing=True)
              for c in COLS})

# 2. NormalizeFeatures: derive targets from inputs, carry the other four through
return {"inputs": e["inputs"].numpy(), "targets": e["inputs"].numpy(),
        **{c: e[c].numpy() for c in COLS[1:]}}

# 3. Pad step: synthesize segmentation/position ONLY when the record didn't provide them
if "inputs_segmentation" not in element:
    ...  # upstream code that computes <col>_segmentation and <col>_position
```

**The 235B run above used a version with parts 1 and 2 but not part 3.** Its carefully built
segmentation was read and then overwritten one step later, so the run trained with cross-document
attention, continuous positions, and no loss mask. Nothing errors and the loss curve looks normal;
we only found it while writing this up. **Check one batch before a long run:**

```python
batch = next(train_iter)
assert batch["inputs_segmentation"].max() > 1, "documents are not separated"
assert (batch["inputs_position"] == 0).sum(axis=1).max() > 1, "positions do not restart"
```

Both checks assume at least some windows in the batch hold more than one document.

**Status as of MaxText 0.2.5 (Oct 2026):** the ArrayRecord path still behaves this way (the file is
now `input_pipeline/input_pipeline_utils.py`). Upstream has since added document-aware
segmentation (`reset_attention_mask=true`, `eod_mask_loss`), but only for Megatron-style `mmap` /
`mmap_npy` data, and it derives boundaries solely from end-of-document tokens. If that's all you
need, converting to that format avoids the patch. Arbitrary loss masks like the one above still
need precomputed `targets_segmentation`, which means the patch.

## 4. XLA / libtpu flags

These overlap collectives with compute (async all-gather, all-to-all, and ragged all-to-all for the
MoE dispatch) and let the latency-hiding scheduler use the full scoped VMEM:

```text
--xla_tpu_dvfs_p_state=7
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
--xla_tpu_enable_async_collective_fusion_multiple_steps=true
--xla_tpu_megacore_fusion_allow_ags=false
--xla_enable_async_collective_permute=true
--xla_enable_async_all_gather=true
--xla_tpu_enable_async_all_to_all=true
--xla_tpu_enable_async_ragged_all_to_all=true
--xla_tpu_enable_ag_backward_pipelining=true
--xla_tpu_overlap_compute_collective_tc=true
--xla_tpu_enable_data_parallel_all_reduce_opt=true
--xla_tpu_data_parallel_opt_different_sized_ops=true
--xla_tpu_scoped_vmem_limit_kib=81920
--xla_tpu_enable_all_experimental_scheduler_features=true
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_scheduler_percent_shared_memory_limit=100
--xla_latency_hiding_scheduler_rerun=2
--xla_lhs_prioritize_async_depth_over_stall=ENABLED
--xla_tpu_aggressive_opt_barrier_removal=ENABLED
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED
--xla_should_add_loop_invariant_op_in_chain=ENABLED
--xla_tpu_host_transfer_overlap_limit=24
--xla_max_concurrent_host_send_recv=100
--xla_tpu_use_minor_sharding_for_major_trivial_input=true
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1
--xla_tpu_assign_all_reduce_scatter_layout=true
--xla_tpu_use_enhanced_launch_barrier=true
--xla_tpu_spmd_rng_bit_generator_unsafe=true
```

We assembled this set by trial and error from flags in Google's published MaxText benchmark
recipes. **Together they changed throughput by only ~0.5–1%.** The parallelism layout and the remat
policy mattered far more, so tune those first and treat the flags as a final polish.

## 5. After training

- Converting the 16 TiB full-state checkpoint to Hugging Face format: [03](03-large-moe-checkpoint-conversion.md).
- Serving it: [04](04-multihost-vllm-inference.md).
