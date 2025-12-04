# Use CLI parameters to define device
import argparse
import asyncio
import gc
import os
import time

import open_clip
import psutil
import torch

gc.enable()

parser = argparse.ArgumentParser(description="OpenCLIP Text Embedding Example")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
args = parser.parse_args()

use_device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {use_device}")


print("Loading model...")
if use_device == "cuda":
    # Reset CUDA memory stats before loading the model
    torch.cuda.reset_peak_memory_stats(use_device)
    torch.cuda.synchronize(use_device)


# Print CPU memory usage before loading the model
process = psutil.Process(os.getpid())
rss_before = process.memory_info().rss
print(f"Memory usage before loading model: {rss_before / (1024 ** 2):.2f} MB")

start_time = time.time()


def load_model():
    model, preprocess = open_clip.create_model_from_pretrained(  # Now scoped within function
        "ViT-g-14",
        "laion2b_s12b_b42k",
        cache_dir="X:\\Code\\miraveja\\models",
        precision="fp16",
        device=use_device,
    )
    print("Model loaded.")

    if use_device == "cuda":
        # Measure GPU memory usage after loading the model
        torch.cuda.synchronize(use_device)
        memory_used = torch.cuda.max_memory_allocated(use_device)
        print(f"GPU memory used by model: {memory_used / (1024 ** 2):.2f} MB")

    # Print CPU memory usage after loading the model
    rss_after = process.memory_info().rss
    print(f"Memory usage after loading model: {rss_after / (1024 ** 2):.2f} MB")
    print(f"Memory used by model: {(rss_after - rss_before) / (1024 ** 2):.2f} MB")

    time.sleep(10)  # Wait for a moment before cleaning up
    print("Cleaning up...")

    del model
    del preprocess


async def main():
    await asyncio.to_thread(load_model)
    gc.collect()

    if use_device == "cuda":
        torch.cuda.empty_cache()

    print("Final memory usage check:")
    rss_final = process.memory_info().rss
    print(f"Final memory usage: {rss_final / (1024 ** 2):.2f} MB")

    loop_start = time.time()
    while True:
        # Keep the script alive for a short while to observe memory usage
        if time.time() - loop_start > 100:  # 100 seconds
            break

    print("Model and preprocess deleted from memory.")


asyncio.run(main())

# I expect that after this point, all memory used by the model is released.

# TEXT = "a photo of a cat"

# # print("Tokenizing and encoding text...")
# # tokens = open_clip.tokenize([TEXT]).to(use_device)

# # print("Generating text embedding...")
# # with torch.no_grad(), torch.amp.autocast(use_device):
# #     embedding = model.encode_text(tokens)
# # print("Text embedding generated.")
# # end_time = time.time()

# print(embedding.shape)  # torch.Size([1, 1536])
# print(f"Time taken to generate text embedding: {end_time - start_time:.2f} seconds")


# del tokens
# del embedding
