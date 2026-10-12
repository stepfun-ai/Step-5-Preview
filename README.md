<h1 align="center">
  <img src="assets/stepfun-logo.png" alt="StepFun logo" width="40">
  Step 5 Preview
</h1>

<p align="center">
  <a href="https://www.stepfun.com/step-5-preview">Model Page</a> &nbsp;|&nbsp;
  <a href="https://platform.stepfun.ai/docs/en/guides/models/step-5-preview">Documentation</a> &nbsp;|&nbsp;
  <a href="https://platform.stepfun.com/">API Platform</a>
</p>

<p align="center">
  <a href="https://www.stepfun.com/step-5-preview"><img src="https://img.shields.io/badge/Model-Step%205%20Preview-00BFA5" alt="Step 5 Preview Model Page"></a>
  <a href="https://platform.stepfun.ai/docs/en/guides/models/step-5-preview"><img src="https://img.shields.io/badge/Docs-Developer%20Guide-2563EB" alt="Developer Documentation"></a>
  <a href="https://platform.stepfun.com/"><img src="https://img.shields.io/badge/API-StepFun-7C3AED" alt="StepFun API Platform"></a>
</p>

## Introduction

Step 5 Preview is StepFun's flagship model for agentic work. It delivers frontier-level performance across software engineering and professional knowledge work, with particular strength in finance.

Step 5 Preview is a sparse Mixture-of-Experts (MoE) model with 600B total parameters and 27B active parameters. It supports up to 1 million tokens of context and accepts text, images, and video, enabling sustained reasoning and tool use across complex tasks.

### Contents

- [Key Capabilities](#key-capabilities)
- [Model Specifications](#model-specifications)
- [Performance](#performance)
- [Deployment](#deployment)
    - [vLLM](#vllm)
    - [Transformers](#transformers)
    - [SGLang](#sglang)
    - [Send a Request](#send-a-request)
- [Resources](#resources)

## Key Capabilities

### Software Engineering

Step 5 Preview supports repository-level software engineering, frontend and visual development, and programming for devices. It can navigate large codebases, coordinate changes across files, run and verify code, and iterate on the results.

### Long-Horizon Agentic Execution

Step 5 Preview can carry out tasks that require repeated tool calls, execution, verification, and revision, drawing on long-context information throughout the workflow.

### Professional Knowledge Work

The model combines research, multimodal understanding, and tool use to analyze long documents, charts, screenshots, images, and video, and produce structured reports and analyses.

### A Focus on Finance

Step 5 Preview is particularly strong in financial tasks that combine quantitative reasoning with analysis of financial data and supporting documents.

## Model Specifications

| Specification | Step 5 Preview |
| --- | --- |
| Architecture | Sparse Mixture-of-Experts (MoE) |
| Total Parameters | 600B |
| Active Parameters | 27B |
| Context Window | Up to 1M tokens |
| Maximum Output | Up to 1M tokens |
| Input | Text, image |
| Output | Text |

## Performance

The following results summarize Step 5 Preview's performance across coding, agentic tasks, professional knowledge work, and finance.

<p align="center">
  <img src="assets/benchmark-overview.png" alt="Step 5 Preview benchmark results" width="100%">
</p>

**Evaluation notes**

- Step 5 Preview results shown above use the **High** reasoning setting.
- GDPval-AA v2.1 scores are based on results from [Artificial Analysis](https://artificialanalysis.ai/) as of September 20, 2026.
- DeepSWE v1.1 evaluations use the [SWE-agent](https://github.com/SWE-agent/SWE-agent) harness with `temperature=1.0` and `top_p=0.95`.

For evaluation context, methodology, and detailed case studies, see the [Step 5 Preview Model Page](https://www.stepfun.com/step-5-preview).

## Deployment

Step 5 Preview can be served through an OpenAI-compatible API with vLLM or SGLang, or used for offline generation with the standalone Transformers implementation. The examples below cover FP8 and BF16 checkpoints.

Use a framework build that includes Step 5 Preview support, including the reasoning, tool-call, and speculative decoding features used in the serving commands.

### H200 Deployment Configurations

| Framework | Precision | Hardware | Parallelism | Interface |
| --- | --- | --- | --- | --- |
| vLLM | FP8 | One H200 node, 8 GPUs | DP2 × TP4; EP8 | OpenAI-compatible API |
| vLLM | BF16 | Two H200 nodes, 16 GPUs | DP4 × TP4; EP16 | OpenAI-compatible API |
| Transformers | FP8 | One H200 node, 8 GPUs | TP8 / EP8 | Offline generation |
| Transformers | BF16 | Two H200 nodes, 16 GPUs | TP16 / EP16 | Offline generation (see validation note below) |

DP, TP, and EP refer to data, tensor, and expert parallelism, respectively. These example configurations are sized for the deployments below. Actual memory requirements, usable context length, and concurrency vary with the checkpoint, inference framework, and KV cache allocation.

### vLLM

#### Environment Setup

We recommend using the official vLLM Docker image:

```bash
docker pull vllm/vllm-openai:latest
```

After pulling the image, follow the [vLLM Docker guide](https://docs.vllm.ai/en/latest/deployment/docker/) to start a GPU-enabled container with access to your model weights. Run the commands below inside the container; the two-node setup also requires network connectivity between the containers.

#### FP8: One H200 Node, 8 GPUs

To serve the FP8 checkpoint on a single 8-GPU H200 node, run the following command. It uses DP2 × TP4 with EP8 and serves the model as `step5`:

```bash
vllm serve stepfun-ai/Step-5-Preview \
    --served-model-name "step5" \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --data-parallel-backend mp \
    --distributed-executor-backend mp \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --max-model-len 1048576 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --spec-method mtp \
    --spec-tokens 3 \
    --reasoning-parser step5 \
    --enable-auto-tool-choice \
    --tool-call-parser step5
```

#### BF16: Two H200 Nodes, 16 GPUs

For the BF16 checkpoint, run one command on each of two 8-GPU H200 nodes. Both nodes must use the same container image and BF16 checkpoint path, accessible inside each container. Set `DP_ADDRESS` to node 0's reachable IP on the high-speed inter-node network, and use the same `DP_RPC_PORT` on both nodes.

On **node 0**, launch DP ranks 0–1 and the API server:

```bash
export MODEL="<STEP5_BF16_MODEL_PATH>"
export DP_ADDRESS="<NODE_0_HIGH_SPEED_NETWORK_IP>"
export DP_RPC_PORT=29550
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 18000 \
  --served-model-name step5 \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --data-parallel-backend mp \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --max-model-len 1048576 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --spec-method mtp \
  --spec-tokens 3 \
  --reasoning-parser step5 \
  --enable-auto-tool-choice \
  --tool-call-parser step5
```

On **node 1**, launch DP ranks 2–3 in headless mode:

```bash
export MODEL="<STEP5_BF16_MODEL_PATH>"
export DP_ADDRESS="<NODE_0_HIGH_SPEED_NETWORK_IP>"
export DP_RPC_PORT=29550
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "$MODEL" \
  --served-model-name step5 \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --data-parallel-backend mp \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --max-model-len 1048576 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --spec-method mtp \
  --spec-tokens 3 \
  --reasoning-parser step5 \
  --enable-auto-tool-choice \
  --tool-call-parser step5 \
  --headless
```

Together, the two nodes run four TP4 groups, with EP16 spanning all 16 GPUs. `--max-num-seqs` applies per DP rank. Only node 0 exposes the API endpoint, on port 18000.

### Transformers

The standalone implementation converts checkpoints into shards for offline generation. With its dependencies installed, run the following commands from the `inference/` directory:

```bash
cd inference
```

Use separate checkpoints and output directories for FP8 and BF16. Conversion preserves the source dtype, and generation runs offline without an API server.

#### FP8: One H200 Node, 8 GPUs

Convert the FP8 checkpoint once, then launch eight local processes for TP8 / EP8 generation:

```bash
export STEP5_FP8_CHECKPOINT="<FP8_CHECKPOINT_DIRECTORY>"
export STEP5_FP8_OUT="<FP8_SHARD_OUTPUT_DIRECTORY>"

python3 convert.py \
  --checkpoint "$STEP5_FP8_CHECKPOINT" \
  --tp-size 8 --ep-size 8 --check-only

python3 convert.py \
  --checkpoint "$STEP5_FP8_CHECKPOINT" \
  --out-dir "$STEP5_FP8_OUT" \
  --tp-size 8 --ep-size 8

torchrun --standalone --nproc-per-node=8 generate.py \
  --tp-dir "$STEP5_FP8_OUT/tp8" --ep-size 8 \
  --prompt "Please introduce tensor parallelism." \
  --max-new-tokens 128
```

#### BF16: Two H200 Nodes, 16 GPUs

For TP16 / EP16 generation across two nodes, first convert the BF16 checkpoint into 16 shards. Make the resulting `tp16/` and `tokenizer_files/` directories accessible to both nodes:

```bash
export STEP5_BF16_CHECKPOINT="<BF16_CHECKPOINT_DIRECTORY>"
export STEP5_BF16_OUT="<BF16_SHARD_OUTPUT_DIRECTORY>"

python3 convert.py \
  --checkpoint "$STEP5_BF16_CHECKPOINT" \
  --tp-size 16 --ep-size 16 --check-only

python3 convert.py \
  --checkpoint "$STEP5_BF16_CHECKPOINT" \
  --out-dir "$STEP5_BF16_OUT" \
  --tp-size 16 --ep-size 16
```

Then run the command below on each node, using the same `MASTER_ADDR`, `MASTER_PORT`, and shard output path. Set `NODE_RANK=0` on node 0 and `NODE_RANK=1` on node 1:

```bash
export STEP5_BF16_OUT="<BF16_SHARD_OUTPUT_DIRECTORY>"
export MASTER_ADDR="<NODE_0_REACHABLE_IP_OR_HOSTNAME>"
export MASTER_PORT=29500
export NODE_RANK=0  # Set to 1 on node 1.

torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  generate.py \
  --tp-dir "$STEP5_BF16_OUT/tp16" --ep-size 16 \
  --prompt "Please introduce tensor parallelism." \
  --max-new-tokens 128
```

This launches 16 processes across the two nodes, matching the 16-shard checkpoint and `--ep-size 16`. The two-node launch uses a shared rendezvous address rather than `--standalone`.

**BF16 validation:** The full BF16 `convert.py → generate.py --tp-dir` file-shard workflow has not yet been validated end to end. The reference implementation has validated checkpoint names and shapes, BF16 small-model conversion, loading and generation, and full-model streaming-weight inference.

### SGLang

#### Environment Setup

Pull the official SGLang Docker image:

```bash
docker pull lmsysorg/sglang:latest
```

After pulling the image, follow the [SGLang installation guide](https://docs.sglang.io/docs/get-started/install) to start a GPU-enabled container with access to your model weights, then run the command below inside it.

#### FP8

The following command serves the FP8 checkpoint as `step5` with TP4 / EP1 and EAGLE speculative decoding. The selected attention and MoE backends require compatible GPU hardware:

```bash
python -m sglang.launch_server \
  --model-path stepfun-ai/Step-5-Preview --served-model-name step5 \
  --tp 4 --ep-size 1 \
  --mem-fraction-static 0.9 --swa-full-tokens-ratio 0.1 \
  --attention-backend fa4 --page-size 64 --disable-prefill-cuda-graph \
  --reasoning-parser step5 --tool-call-parser step5 \
  --moe-runner-backend flashinfer_trtllm \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle --max-running-requests 64 \
  --chunked-prefill-size 16384 \
  --flashinfer-allreduce-fusion-backend trtllm
```

### Send a Request

Once the server is ready, set `BASE_URL` for your deployment:

| Serving configuration | Base URL on the server host |
| --- | --- |
| vLLM FP8 | `http://localhost:8000` |
| vLLM BF16 | `http://localhost:18000` (node 0) |
| SGLang FP8 | `http://localhost:30000` |

The example below targets vLLM FP8:

```bash
export BASE_URL="http://localhost:8000"

curl "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step5",
    "messages": [
      {"role": "user", "content": "Please introduce tensor parallelism."}
    ]
  }'
```

## Resources

- [Step 5 Preview Model Page](https://www.stepfun.com/step-5-preview) — Model overview, evaluation context, and long-horizon case studies.
- [Documentation](https://platform.stepfun.ai/docs/en/guides/models/step-5-preview) — Official documentation and developer guides.
- [API Platform](https://platform.stepfun.com/) — Model access and platform services.
- [StepFun Website](https://www.stepfun.com/) — Company information and model announcements.
