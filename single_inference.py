"""
This script measures the latency and throughput of singular inference requests.
"""
import os
import time
import statistics
from dotenv import load_dotenv
import litellm


load_dotenv()

LOCAL_API_BASE = os.getenv("LOCAL_API_BASE")

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

prompts = [
    "Hello, how is the weather today?",
    "What is the capital of France?",
    "What is the meaning of life?",
    "What is the capital of the moon?",
    "What is the capital of the sun?",
]

start_time = time.time()
times = []
for idx, prompt in enumerate(prompts, 1):
    query_start = time.time()
    response = litellm.completion(
        model = "openai/Qwen/QwQ-32B-Preview",
        messages = [{ "content": prompt, "role": "user"}],
        api_key = "nokeyneeded",
        api_base = LOCAL_API_BASE,
        num_retries = 3
    )
    query_time = time.time() - query_start
    times.append(query_time)
    print(f"Query {idx} time: {query_time:.2f} seconds")

total_time = time.time() - start_time

# Calculate statistics
avg_time = statistics.mean(times)
median_time = statistics.median(times)
std_dev = statistics.stdev(times) if len(times) > 1 else 0 # pylint: disable=invalid-name
min_time = min(times)
max_time = max(times)

print("\nStatistics:")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per query: {avg_time:.2f} seconds")
print(f"Median time: {median_time:.2f} seconds")
print(f"Standard deviation: {std_dev:.2f} seconds")
print(f"Min time: {min_time:.2f} seconds")
print(f"Max time: {max_time:.2f} seconds")
print(f"Throughput: {len(prompts)/total_time:.2f} queries/second")
