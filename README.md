# parallel_inference_demo
An educational repository showcasing the benefits of parallel processing.

## Setup

1. Use the `environment.yml` file to create a conda environment with the necessary dependencies.

```bash
conda env create -f environment.yml
```

2. Copy the `.env.example` file to `.env` and set the `LOCAL_API_BASE` variable to the URL of the SGLang server. Alternatively, use the OpenAI API.

## Usage

Activate the conda environment.

```bash
conda activate parallel_inference_demo
```

Start an SGLang Server in a terminal. Here is an example command:

```
CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server --model-path Qwen/QwQ-32B-Preview --port 30010 --host 0.0.0.0 --tp-size 2
```

This starts an OpenAI-compatible server on port 30010 using a Qwen model, sharded across 2 GPUs.

## Running the scripts

```bash
python single_inference.py # check latency and throughput of single requests
python parallel_inference.py # check latency and throughput of parallel requests
python optimal_inference.py # check latency and throughput of a large set of parallel requests (will not complete!!)
```

