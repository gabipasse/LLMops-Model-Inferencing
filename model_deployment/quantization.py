import torch
import time
import random
import typing
import ctranslate2
from torch import device, cuda, Tensor
from tqdm.auto import tqdm
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from datasets import load_dataset
from ctranslate2.converters import TransformersConverter


@contextmanager
def track_time() -> typing.Generator[None, None, None]:
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


def batch_generate(tokens: Tensor, tokenizer: GPT2TokenizerFast, model: GPT2LMHeadModel) -> list[str]:
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
        print(f"Shape of tokens: {tokens.shape}")
        # num_beams is set to two in order to enforce the same behavior and comparable executions between quantized and base pytorch model
        outputs = model.generate(
            tokens,
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=2,
            repetition_penalty=1.5,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Pegar max size de acordo com maior len de input, para evitar que algum input nao seja encoded e decoded
def predict_sorted_batches(prompts: list[str], tokenizer: GPT2TokenizerFast, model: GPT2LMHeadModel, avaible_device: device) -> typing.Generator[list[str], None, str]:
    """Applies sorting and dynamic batching techniques to inference from inputs.

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

    # Tensor format is removed because tensors require inputs of the same length (I removed it because I removed padding)
    tokenized_inputs = tokenizer(
        prompts, padding=False, truncation=True, max_length=128
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
    max_size = max(sorted_batches.keys())

    for length, sorted_batch in sorted_batches.items():
        print("length:", length)
        current_batch = []
        current_batch_size = 0
        for batch in sorted_batch:
            if current_batch_size + len(batch) > max_size:
                print("Tamanho em tokens:", current_batch_size)
                tensor_batch = torch.tensor(current_batch).to(avaible_device)
                yield batch_generate(tensor_batch, tokenizer, model)
                del tensor_batch
                current_batch, current_batch_size = [], 0

            current_batch.append(batch)
            current_batch_size += len(batch)

        if current_batch:
            tensor_batch = torch.tensor(current_batch).to(avaible_device)
            yield batch_generate(tensor_batch, tokenizer, model)
            del tensor_batch


# CTranslate2 batching with quantized model
def batch_generate_using_ctrans(prompts: list[str], tokenizer: GPT2TokenizerFast, generator_model: ctranslate2._ext.Generator, max_batch_size: int=4) -> list[str]:
    """Use quantized models to inference from inputs.

    Parameters
    ----------
    prompts: list[str]
        List of inputs to be treated as batches and inferenced from.
    tokenizer: GPT2TokenizerFast
        Tokenizer which will be used to tokenize the string input into tokenized indexed values.
    generator_model: ctranslate2._ext.Generator
        Quantized model which will be used for inference.
    max_batch_size: int=4
        Maximun amount of inputs in each batch.

    Return
    ------
    list[str]
        Response generated by the model.
    """

    # Padding is not defined because CTranslate2 implements sorting batching by default
    all_results = []
    for i in range(0, len(prompts), max_batch_size):
        batch = prompts[i : i + max_batch_size]
        inputs = [
            tokenizer.tokenize(prompt, truncation=True, max_length=128)
            for prompt in batch
        ]
        max_batch_size = max(len(input_tokens) for input_tokens in inputs)

        results: list[ctranslate2._ext.GenerationResult] = generator_model.generate_batch(
            inputs,
            max_length=128,
            max_batch_size=max_batch_size,
            beam_size=2,
            repetition_penalty=1.5,
        )

        # Results contains 3 lists of lists: sequence_ids, scores, attention_weights
        # Change here: Access the generated IDs directly from the result object
        results_ids = [res.sequences_ids[0] for res in results]
        all_results.extend(tokenizer.batch_decode(results_ids, skip_special_tokens=True))

    return all_results


def main():

    dataset = load_dataset("hakurei/open-instruct-v1", split="train")
    example_prompts_4_sorting_prediction = random.sample(dataset["instruction"], k=300)

    avaible_device = device("cuda" if cuda.is_available() else "cpu")
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        "TheFuzzyScientist/diabloGPT_open-instruct"
    ).to(avaible_device)
    model.eval()
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-medium", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    input_path = "models/gpt-instruct"
    # Convert the model to CTranslate2
    model.save_pretrained(input_path)
    tokenizer.save_pretrained(input_path)

    converter = TransformersConverter(input_path)
    output_path = converter.convert(
        output_dir="models/gpt-instruct-quant", quantization="float16", force=True
    )

    generator_model = ctranslate2._ext.Generator(output_path, device=avaible_device.type)

    with track_time():
        generator_sorted_batches = predict_sorted_batches(example_prompts_4_sorting_prediction, tokenizer, model, avaible_device)
        try:
            for batch_prediction in tqdm(generator_sorted_batches):
                print("Amount of inputs in this batch", len(batch_prediction))
        except StopIteration as e:
            print(f"Generator returned: {e.value}")

    with track_time():
        generator_ctrans_method = batch_generate_using_ctrans(example_prompts_4_sorting_prediction, tokenizer, generator_model)
        try:
            for batch_prediction in tqdm(generator_ctrans_method):
                print("Amount of inputs in this batch", len(batch_prediction))
        except StopIteration as e:
            print(f"Generator returned: {e.value}")


if __name__ == "__main__":
    main()
