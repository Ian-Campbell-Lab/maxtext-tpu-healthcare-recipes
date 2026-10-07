# Installing MaxText, Tunix and vLLM-TPU without public internet

Our TPU VMs and GKE nodes have **no route to pypi.org, github.com, or huggingface.co**. They can
reach Google APIs (Cloud Storage, Artifact Registry) through
[Private Google Access](https://cloud.google.com/vpc/docs/private-google-access). This recipe is how
we installed the MaxText post-training stack under that constraint.

The key fact: **Google publishes Artifact Registry Python repositories that carry the JAX/TPU
dependency tree.** Because Artifact Registry is a Google API, those indexes stay reachable when PyPI
is blocked. Everything they don't carry, you stage in a Cloud Storage bucket from a machine that
does have internet access.

> Last run: Python 3.12, MaxText `>=0.2.0` from the package indexes (Mar 2026), and the source
> `setup.sh` path (Sep–Oct 2025). Package versions move quickly; treat the pins as examples.

Placeholders: `BUCKET` (staging bucket), `INTERNAL_INDEX` (your institution's own Python mirror, if
it has one; delete those lines if not).

## 0. Make Google APIs resolvable (VMs without external IPs)

If DNS on your VMs doesn't already send Google API hostnames to the Private Google Access VIP, map
them yourself before anything else. `199.36.153.8–11` is `private.googleapis.com`:

```bash
# Add the <location>-python.pkg.dev / <location>-apt.pkg.dev hosts your registries use.
for h in storage.googleapis.com packages.cloud.google.com us-python.pkg.dev; do
  echo "199.36.153.8 $h" | sudo tee -a /etc/hosts
done
```

apt needs the same treatment if you install system packages: replace `/etc/apt/sources.list` with
Artifact Registry apt remote repositories (or your institution's mirror) and remove the default
Ubuntu sources, which will otherwise time out.

## 1. Stage what the indexes don't carry

From a machine with internet access, copy these to `gs://BUCKET/staging/`:

- A relocatable Python build, e.g. a
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone) `install_only`
  tarball for 3.12. The TPU VM base image's Python is older than MaxText needs.
- **Git-only dependencies.** At the time, MaxText pulled `qwix` from GitHub. Stage it as a bare repo
  tarball (`git clone --bare https://github.com/google/qwix qwix.git && tar czf qwix.git.tar.gz qwix.git`).
- For the source-install path only: every `@ https://github.com/.../archive/<sha>.zip` dependency
  in MaxText's requirements files, downloaded as `<sha>.zip`.
- Any model or tokenizer files your workflow fetches from the Hugging Face Hub (see
  [03](03-large-moe-checkpoint-conversion.md) — the checkpoint converter does this).

## 2. Python and a virtual environment

```bash
gcloud storage cp gs://BUCKET/staging/cpython-3.12.*-x86_64-unknown-linux-gnu-install_only.tar.gz .
sudo mkdir -p /opt/python-3.12
sudo tar -xf cpython-3.12.*-install_only.tar.gz -C /opt/python-3.12 --strip-components=1

# On TPU VMs, set this before activating, or libpython is not found.
export LD_LIBRARY_PATH=/opt/python-3.12/lib:$LD_LIBRARY_PATH
/opt/python-3.12/bin/python3.12 -m venv ~/venvs/maxtext-312
source ~/venvs/maxtext-312/bin/activate
```

## 3. Point pip and uv at the Google-hosted indexes

Artifact Registry authenticates through the VM's service account via a keyring plugin. Install it
from the public JAX index first:

```bash
pip install uv keyring keyrings.google-artifactregistry-auth \
  --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
```

```bash
mkdir -p ~/.config/pip ~/.config/uv
cat > ~/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
extra-index-url = https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/
                  https://INTERNAL_INDEX/simple/
EOF

cat > ~/.config/uv/uv.toml <<'EOF'
[[index]]
name = "maxtext-rl"
url = "https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/"
default = true          # replaces PyPI as the default index

[[index]]
name = "jax"
url = "https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"

[[index]]
name = "internal"
url = "https://INTERNAL_INDEX/simple/"
authenticate = "always"
EOF
```

## 4. Redirect git dependencies to the staged copies

```bash
gcloud storage cp -r gs://BUCKET/staging/ ~/staging
cd ~/staging && tar -xzf qwix.git.tar.gz
git config --global url."$HOME/staging/qwix.git".insteadOf https://github.com/google/qwix
```

`insteadOf` rewrites the URL for every tool that shells out to git, including pip and uv resolving a
`git+https://` requirement, so nothing in MaxText needs editing.

## 5a. Install from the package indexes (MaxText ≥ 0.2.0) — recommended

```bash
uv pip install "maxtext[tpu]>=0.2.0"            --resolution=lowest
uv pip install "maxtext[tpu-post-train]>=0.2.0" --resolution=lowest
uv pip install "maxtext[runner]"                --resolution=lowest
uv pip install vllm-tpu
```

`--resolution=lowest` came from an upstream recipe at the time. As of Oct 2026 it isn't needed:
`maxtext[tpu]` resolves at default resolution (to 0.2.5; with `lowest`, to 0.2.0).

**Install vLLM-TPU in a separate command, or a separate environment.** As of Oct 2026, every
`maxtext[tpu]>=0.2.0` requires `setuptools>=80.9` and every `vllm-tpu` pins `setuptools==78.1.0`,
so resolving them together fails. Installing them one after the other, as above, worked for us. The
second install downgrades `setuptools`, and `uv pip check` will then report the mismatch. If you'd
rather not carry that, keep separate virtual environments for training and serving.

## 5b. Install from source with `setup.sh` (older MaxText)

Before MaxText shipped as a package, `setup.sh` reached out to GitHub and to a public GCS
`find-links` page. With the source tree staged (`gs://BUCKET/staging/maxtext`), rewrite those
references:

```bash
cd ~/maxtext
# GitHub archive pins -> staged zips
for f in requirements.txt src/install_maxtext_extra_deps/extra_deps_from_github.txt \
         generated_requirements/tpu-requirements.txt; do
  [ -f "$f" ] && sed -E -i \
    "s|@ https://github.com/[^ ]*/archive/([0-9a-f]{7,40})\.zip|@ file://$HOME/staging/\1.zip|g" "$f"
done
# Let the Artifact Registry indexes resolve libtpu instead of the public find-links page
sed -i 's|-f https://storage.googleapis.com/jax-releases/libtpu_releases.html||g' setup.sh
# Install tunix from the index rather than from git
sed -i "s|['\"]git+https://github.com/google/tunix.git['\"]|tunix|g" setup.sh

bash setup.sh MODE=nightly DEVICE=tpu
```

## 6. Smoke test

```python
from vllm import LLM, SamplingParams
llm = LLM("/path/to/any/hf/model", tensor_parallel_size=8,
          max_model_len=2048, enable_prefix_caching=False, enforce_eager=True)
print(llm.generate(["How does a rainbow work?"], SamplingParams(max_tokens=256))[0].outputs[0].text)
```

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the environment. Without them, a missing
local file shows up as a network timeout rather than as a clear "file not found" error.
