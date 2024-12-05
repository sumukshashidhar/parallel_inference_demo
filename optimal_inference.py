"""
This script measures the latency and throughput of a large set of parallel inference requests.
"""
import os
import time
import uuid
from dotenv import load_dotenv
import litellm

load_dotenv()

LOCAL_API_BASE = os.getenv("LOCAL_API_BASE")

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

prompts = [
    f"{prompt} - Also think about {uuid.uuid4()}"
    for prompt in [
        "Hello, how is the weather today?",
        "What is the capital of France?",
        "What is the meaning of life?",
        "What is the capital of the moon?",
        "What is the capital of the sun?",
    ]
] * 5000

start_time = time.time()
responses = litellm.batch_completion(
    model="openai/Qwen/QwQ-32B-Preview",
    messages=[[{"role": "user", "content": prompt}] for prompt in prompts],
    api_base=LOCAL_API_BASE,
    api_key="nokeyneeded",
    num_retries=3,
)
total_time = time.time() - start_time

# Calculate statistics
avg_time = total_time / len(prompts)

print("\nStatistics:")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per query: {avg_time:.2f} seconds")
