import asyncio
import os
import typing as t
from dotenv import load_dotenv
from datasets import load_dataset

from openai import OpenAI

from chromadb.utils import embedding_functions

load_dotenv()

COLLECTION_NAME = "code_embeddings"
EMB_MODEL_NAME = 'all-MiniLM-L6-v2'  # A good general-purpose model, consider code-specific models for better results

"""
By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings. 
This embedding model can create sentence and document embeddings that can be used for a wide variety of tasks. 
This embedding function runs on local machine, and script may download the model files (this will happen automatically).
The cache location for the model is ~/.cache/chroma/
"""
default_ef = embedding_functions.ONNXMiniLM_L6_V2()
print(default_ef.name())
print(default_ef.DOWNLOAD_PATH)

p1 = "Find the number of triples of nonnegative integers (a,b,c) satisfying a " \
     "+ b + c = 300 and a^2b + a^2c + b^2a + b^2c + c^2a + c^2b = 6,000,000."

p2 = "Let $ABC$ be a triangle inscribed in circle $\\omega$. Let the tangents to $\\omega$ at $B$ and $C$ intersect " \
     "at point $D$, and let $\\overline{AD}$ intersect $\\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be " \
     "written as the form $\\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$."


async def req():
    client = OpenAI()
    print("OpenAI client initialized successfully!")

    # Continuing from the setup above...

    print("\n--- Example 1: Text-Only Chat ---")

    try:
        msg = [
            {"role": "system", "content": "You are a helpful assistant that solves complex problems."},
            {"role": "user", "content": p2 }
        ]
        length = len(str(msg))
        response = client.chat.completions.create(
            model="o3-mini",  # Choose your model
            messages=msg,
            max_completion_tokens=8000  # Limit the length of the response
        )

        # Extracting and printing the response content
        ai_response = response.choices[0].message.content
        print(f"AI Response: {ai_response}")
        print(f"input bytes: {length}")
        print(f"prompt_tokens: {response.usage.prompt_tokens}")
        print(f"completion_tokens: {response.usage.completion_tokens}")


    except Exception as e:
        print(f"An error occurred: {e}")


async def li_models():
    client = OpenAI()
    print("OpenAI client initialized successfully!")

    models = client.models.list()
    for model in models:
        print(model.id)


async def iter_ds():
    ds = load_dataset(path="HuggingFaceH4/aime_2024", name="default")

    print(ds.shape)
    p = dict(ds['train'][1])
    for k, v in p.items():
        print(f"{k}:")
        print(v)


async def main():
    test_str = "lets x=10"
    emb = default_ef.embed_with_retries([test_str])
    print(emb[0].shape)
    # await li_models()
    await req()
    # await iter_ds()


if __name__ == "__main__":
    asyncio.run(main())
