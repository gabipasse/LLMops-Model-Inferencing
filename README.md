# LLMops Model Inferencing
![Static Badge](https://img.shields.io/badge/PyTorch-%23ffffff?style=for-the-badge&logo=PyTorch&logoColor=black&labelColor=%23EE4C2C&color=white)
![Static Badge](https://img.shields.io/badge/HuggingFace-%23ffffff?style=for-the-badge&logo=HuggingFace&logoColor=black&labelColor=%23FFD21E&color=white)
![Static Badge](https://img.shields.io/badge/Transformers-%23ffffff?style=for-the-badge&logo=HuggingFace&logoColor=black&labelColor=%23FFD21E&color=white)
![Static Badge](https://img.shields.io/badge/Typing-%23ffffff?style=for-the-badge&logo=Python&logoColor=black&labelColor=%233776AB&color=white)
![Static Badge](https://img.shields.io/badge/ctranslate2-%23ffffff?style=for-the-badge&logo=Python&logoColor=black&labelColor=%233776AB&color=white)

This project explores techniques for optimizing the inference process using LLMs and SLMs through batching. The methods discussed are useful for systems where the availability of computational resources, latency, and throughput are critical.

Going beyond a model's accuracy, real-world applications based on language models depend on the ability of these models to adapt to user requirements and contexts. A model with high accuracy (calculated, for example, by cosine similarity) but with very high latency can be useless in contexts where near-real-time inference is crucial.

As always, the first step in deciding whether to use LLM/SLM-based solutions is to assess the availability of less costly alternatives that meet the requirements expressed by clients and/or the context. If there are solutions based on classic machine learning models that are less expensive than those required for the same need with LLMs/SLMs, classic models should be preferred.

## MLOps Best Pratices
The different inference techniques have been divided into files to modularize the code and facilitate the understanding of the execution flow for each method. Defined and used functions have docstrings to simplify debugging and code refactoring. Type hints are used for the same purpose. The latter is extremely important because some objects are instantiated from classes with the same name but from different modules; this is the case with "Generator," which is present in both the "ctranslate" and "typing" modules.

## Dynamic Batching
Dynamic batching is a batching technique that defines the number of inputs passed in each batch (for inference) based on the number of tokens present in each batch, rather than the number of inputs themselves. This technique addresses the following issue: if a value x is set for the number of inputs to be passed in each batch without considering the size of each input, it can result in batches with extremely different sizes. This can lead to both underutilization of computational resources and memory overflow on the GPU/CPU.

## Sorting Batching

Sorting batching is a batching inference method in which the inputs are not tokenized into tensors. This approach is important because, for tensors to be passed in batches for a model to perform the inference process, they need to have the same length. This means that all inputs will have their length differences compensated by the insertion of padding tokens. The presence of these tokens increases the computational requirements of the inference process and can also affect the model's accuracy (although models based on GPT-2 are trained to reduce the impact of padding tokens—primarily by applying padding to the left, as it is an autoregressive model).

## Sorting and Dynamic Batching em Conjunto

By combining the sorting and dynamic batching techniques, the inference process is further optimized. Merging these two techniques allows each batch to be defined according to the length of the combination of inputs passed as a batch. At the same time, by defining the outputs of tokenization without the need for padding, the size of each input is known. This approach also simplifies the understanding of the amount of computational resources that will be required for the inference of a given number of inputs x.

## Quantização

Model quantization is the reduction of precision in the results of operations performed during the training and/or inference processes. As highlighted regarding the importance of seeking the most optimal trade-off between accuracy and other requirements of language model-based applications, this technique can affect the model's accuracy but can, in turn, reduce latency and the computational resources required. The ctranslate2 library implements model quantization and also utilizes the combination of dynamic and sorting batching techniques by default

> "I Believe Because it is Absurd."
> — Misinterpretation of Tertullian's Credo