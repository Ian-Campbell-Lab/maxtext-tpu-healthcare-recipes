# MaxText on Cloud TPU in a locked-down healthcare environment

Recipes from training and serving large language models on Cloud TPUs with
[MaxText](https://github.com/AI-Hypercomputer/maxtext) and JAX inside a hospital research
environment. In that environment the compute has **no public internet access**: no PyPI, no GitHub,
no Hugging Face Hub. Everything runs in a single Google Cloud project on GKE and TPU VMs.



The models were continued-pretrained on clinical text at Children's Hospital of Philadelphia
(CHOP). This repository documents infrastructure and methodology used in clinical-model research.
It contains no patient data, clinical-text samples, patient identifiers, model weights, checkpoints,
dataset manifests, or code configured to access institutional patient-data systems. It contains
only the infrastructure recipes, which apply to any domain.

| # | Recipe | What it adds to the upstream docs |
|---|---|---|
| 01 | [Installing MaxText, Tunix and vLLM-TPU without public internet](recipes/01-airgapped-install.md) | Google-hosted Artifact Registry indexes as a PyPI replacement; redirecting git and GitHub-archive dependencies to staged copies |
| 02 | [Continued pretraining of Qwen3-235B-A22B on 64 v7x chips with MaxText on GKE](recipes/02-moe-continued-pretraining-v7x-gke.md) | A complete, measured config (~143B tokens in 11 days) for a 128-expert MoE on Ironwood, run as a plain Indexed Job |
| 03 | [Converting a 235B MoE checkpoint to Hugging Face format](recipes/03-large-moe-checkpoint-conversion.md) | Host-memory sizing that worked ([MaxText #3418](https://github.com/AI-Hypercomputer/maxtext/issues/3418)), and the converter's Hub dependency when the Hub is unreachable |
| 04 | [Serving a 235B model with vLLM-TPU across a multi-host slice](recipes/04-multihost-vllm-inference.md) | Ray across TPU VM workers; streaming weights from Cloud Storage through a RAM-backed gcsfuse cache |

**PyTorch on TPU, too.** The embedding stage of our health-system-scale semantic search (484M text
chunks embedded on Trillium TPUs with PyTorch/XLA) is published separately as a reference
implementation:
[clinical-semantic-search](https://github.com/Ian-Campbell-Lab/clinical-semantic-search) (see
`etl/parallel_embedding.py`).

Each recipe states when it was last run and with which versions. MaxText changes quickly, so the
flags and paths are a record of what worked, not a supported interface. Values written as
`UPPER_CASE` are placeholders for your own project, buckets, and accounts.

## Credits

The checkpoint conversion, container build, and GKE Pathways setup are the work of Faith Mutinda.

## Acknowledgments

Ironwood (v7x) capacity was provided by Google through its private preview program. Trillium
(v6e) capacity was provided through the Google
[TPU Research Cloud](https://sites.research.google/trc/). This work is supported by a gift from
Google's TPU Builders program.

## License

Apache 2.0, matching MaxText. See [LICENSE](LICENSE).
