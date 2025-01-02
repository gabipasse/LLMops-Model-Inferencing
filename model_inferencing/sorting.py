import torch
import time
import random
from torch import device, cuda, Tensor
from tqdm.auto import tqdm
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from datasets import load_dataset
from typing import Generator


@contextmanager
def track_time() -> Generator[None, None, None]:
    """Tracks the time taken to inference from a batch of inputs.

    Parameters
    ----------

    Yield
    -----
    Generator[None, None, None]
        Generator used to track the time of inference"""
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.2f}s")


# Normal batching
def chunker(
    seq: list[list[int]], max_size: int
) -> list[Generator[list[str], None, str]]:
    """Split the inputs considering the maximun size supported by the avaible_device.

    Parameters
    ----------
    seq: list[list[int]], size: int
        Sequence of inputs to be chunked considering the max_size.
    max_size: int
        Nax size of each chunk of inputs.

    Return
    ------
    list[Generator[list[str], None, str]]
        Chunk of inputs considering the max_size.
    """

    return (seq[pos : pos + max_size] for pos in range(0, len(seq), max_size))


def batch_generate(
    tokens: Tensor, tokenizer: GPT2TokenizerFast, model: GPT2LMHeadModel
) -> list[str]:
    """Generate predictions from tokenized inputs, treated as batches.

    Parameters
    ----------
    tokens: Tensor
        Tokenized inputs to be treated as batches and inferenced from.
    tokenizer: GPT2TokenizerFast
        Tokenizer which will be used to tokenize the string input into tokenized indexed values.
    model: GPT2LMHeadModel
        Model which will be used for inference.

    Return
    ------
    list[str]
        Response generated by the model.
    """

    # max_length method is the sum of the input length + output length
    # max_new_tokens is the sum of the output length
    if tokens.ndim == 1:
        print(f"Shape of tokens before unsqueeze: {tokens.shape}")

        # Add a dimension at the beginning to make it a 2D tensor
        tokens = tokens.unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(
            tokens, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Sorted batching
def predict_sorted_batches(
    prompts: list[str],
    max_batch_size: int,
    tokenizer: GPT2TokenizerFast,
    model: GPT2LMHeadModel,
    avaible_device: device,
) -> Generator[list[str], None, str]:
    """Applies sorting batching technique to inference from inputs.

    Parameters
    ----------
    prompts: list[str]
        List of inputs to be treated as batches and inferenced from.
    max_tokens: int
        Maximun amount of token to be processed within the same batch.
    tokenizer: GPT2TokenizerFast
        Tokenizer which will be used to tokenize the string input into tokenized indexed values.
    model: GPT2LMHeadModel
        Model which will be used for inference.
    avaible_device: device
        Device (CPU or GPU) on which the model's predictions and tokenization will be performed.

    Yield
    ------
    Generator[list[str], None, str]
        Response generated by the model.
    """

    # Tensor format is removed because tensors require inputs of the same length (It is removed because padding is disabled)
    tokenized_inputs = tokenizer(
        prompts, padding=False, truncation=True, max_length=512
    )["input_ids"]

    sorted_tokens = sorted(tokenized_inputs, key=len)
    sorted_batches: dict[int, list[list[int]]] = {}
    for sorted_token in sorted_tokens:
        if not len(sorted_token):
            continue

        length = len(sorted_token)
        if length not in sorted_batches:
            sorted_batches[length] = []

        sorted_batches[length].append(sorted_token)

    for length, sorted_batch in sorted_batches.items():
        for batch in chunker(sorted_batch, max_batch_size):
            tensor_batch = torch.tensor(batch).to(avaible_device)
            print("Length of each input in this batch", length)
            yield batch_generate(tensor_batch, tokenizer, model)
            del tensor_batch

    return "All batches have been processed."


def main():
    avaible_device = device("cuda" if cuda.is_available() else "cpu")
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        "TheFuzzyScientist/diabloGPT_open-instruct"
    ).to(avaible_device)
    model.eval()

    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-medium", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("hakurei/open-instruct-v1", split="train")
    example_prompts = dataset["instruction"][-4:]
    example_tokenized_inputs = tokenizer(
        example_prompts, return_tensors="pt", padding=True
    ).to(avaible_device)["input_ids"]

    print("example tokenized inputs", example_tokenized_inputs)

    # This is to demonstrate how padding in this manner introduces several padding tokens into the sentences,
    # which affects the performance of the model's attention mechanism.
    # Additionally, since these padding tokens pass through the model, they are also copied to the GPU, embedded, and decoded...
    # (this is also a matter of computational efficiency).
    print(
        "\n\n".join(tokenizer.batch_decode(example_tokenized_inputs)).replace(
            tokenizer.eos_token, "[PAD]"
        )
    )
    example_prompts_4_sorting_prediction = random.sample(dataset["instruction"], k=300)

    with track_time():
        generator = predict_sorted_batches(
            example_prompts_4_sorting_prediction, 32, tokenizer, model, avaible_device
        )
        try:
            for batch_prediction in tqdm(generator):
                print("Amount of inputs in this batch:", len(batch_prediction))
        except StopIteration as e:
            print(f"Generator returned: {e.value}")


if __name__ == "__main__":
    main()
