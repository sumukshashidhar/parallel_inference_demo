# Parallel Inference Demo

A comprehensive demonstration of parallel processing strategies for Large Language Model (LLM) inference, showcasing how different approaches affect latency and throughput.

## 🎯 Purpose

This repository serves as an educational resource to understand and compare different inference strategies when working with LLMs:

- Single sequential requests
- Parallel batch processing
- Large-scale parallel inference

The demo uses the Qwen model family as an example, but the concepts apply to any LLM deployment.

## 🏗️ Architecture

The project consists of three main components:

1. **Single Inference** (`single_inference.py`)
   - Processes requests one at a time
   - Measures individual request latency
   - Calculates basic throughput metrics
   - Provides detailed statistics (mean, median, std dev)

2. **Parallel Inference** (`parallel_inference.py`)
   - Processes multiple requests simultaneously
   - Uses LiteLLM's batch completion feature
   - Demonstrates basic parallel processing benefits

3. **Optimal Inference** (`optimal_inference.py`)
   - Stress tests the system with 5000 parallel requests
   - Shows real-world scaling behavior
   - Demonstrates system limitations

## 🚀 Getting Started

### Prerequisites

- CUDA-capable GPU(s)
- Conda package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parallel_inference_demo.git
cd parallel_inference_demo
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate parallel_inference_demo
```

3. Configure the environment:
```bash
cp .env.example .env
```
Edit `.env` to set `LOCAL_API_BASE` to your SGLang server URL.

### Starting the Inference Server

Launch the SGLang server with tensor parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path Qwen/QwQ-32B-Preview \
    --port 30010 \
    --host 0.0.0.0 \
    --tp-size 2
```

This creates an OpenAI-compatible endpoint running the Qwen model across 2 GPUs.

## 📊 Running Experiments

1. Test single request performance:
```bash
python single_inference.py
```
This will show baseline performance metrics for sequential processing.

2. Test parallel request performance:
```bash
python parallel_inference.py
```
This demonstrates the benefits of basic parallel processing.

3. Test large-scale parallel processing:
```bash
python optimal_inference.py
```
Note: This script intentionally pushes system limits and may not complete execution.

## 📈 Key Metrics

The demo measures and reports:

- Total processing time
- Average time per query
- Median response time (for single inference)
- Standard deviation (for single inference)
- Throughput (queries per second)

## 🔧 Technical Details

- Uses LiteLLM for standardized model interaction
- Integrates with Langfuse for observability
- Supports tensor parallelism via SGLang
- Handles both CPU and GPU acceleration
- Implements proper error handling and retries

## 📝 Notes

- The optimal_inference script is designed to demonstrate system limits and is not meant for production use
- Performance will vary based on hardware, model size, and input complexity
- The demo uses a fixed set of test prompts for consistency

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements to:

- Additional inference strategies
- Better performance metrics
- Documentation improvements
- Bug fixes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

