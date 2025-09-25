#!/usr/bin/env python3
""" Common utilities for Stable Diffusion XL operations """

import torch
from diffusers import DiffusionPipeline
from pipeline_utilities import logging_utils

# Module-level logger
logger = logging_utils.get_logger(__name__)


# Device constants
class Devices:
    """ Constants for supported device types """
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


# Valid device types for SDXL operations
VALID_DEVICES = {Devices.CUDA, Devices.MPS, Devices.CPU}


def get_device() -> str:
    """ Select the best available compute device: MPS (Apple), CUDA (NVIDIA), or CPU fallback """
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) for acceleration")
        return Devices.MPS
    if torch.cuda.is_available():
        logger.info("Using CUDA for acceleration")
        return Devices.CUDA
    logger.info("Using CPU (no GPU acceleration available)")
    return Devices.CPU


def get_optimal_dtype(device: str) -> torch.dtype:
    """ Get the optimal data type for the given device """
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Must be one of: {', '.join(VALID_DEVICES)}")

    if device == Devices.CUDA:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def optimize_pipeline(pipeline: DiffusionPipeline, device: str) -> None:
    """ Apply standard memory and compute optimizations to a diffusion pipeline """
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Must be one of: {', '.join(VALID_DEVICES)}")

    logger.debug("Optimizing pipeline for device: %s", device)

    # Enable CPU offloading for CUDA devices to save VRAM
    if device == Devices.CUDA:
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            logger.debug("Enabled CPU offloading for CUDA device")

    # Enable attention slicing to reduce memory usage
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
        logger.debug("Enabled attention slicing for memory optimization")


def create_generator(device: str, seed: int) -> torch.Generator:
    """ Create a seeded random generator for the specified device """
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Must be one of: {', '.join(VALID_DEVICES)}")

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got: {seed}")

    logger.debug("Creating generator for device %s with seed %d", device, seed)
    return torch.Generator(device=device).manual_seed(seed)


def _split_tokens_into_chunks(tokens: list[int], max_tokens: int) -> list[list[int]]:
    """ Split a list of token IDs into chunks of specified maximum size """

    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got: {max_tokens}")

    if not isinstance(tokens, list):
        raise TypeError(f"tokens must be a list, got: {type(tokens)}")

    if not tokens:
        return []

    chunks = []
    current_chunk = []

    for token in tokens:
        if not isinstance(token, int):
            raise TypeError(f"All tokens must be integers, got: {type(token)}")

        if len(current_chunk) >= max_tokens:
            chunks.append(current_chunk)
            current_chunk = [token]
        else:
            current_chunk.append(token)

    # Add the last chunk if it has tokens
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def encode_long_prompt(pipeline: DiffusionPipeline, prompt: str, max_tokens: int = 75) -> torch.Tensor:
    """ Encode a text prompt that exceeds the standard 77 token limit using chunking and concatenation """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got: {type(prompt)}")

    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got: {max_tokens}")

    if not hasattr(pipeline, 'encode_prompt'):
        raise TypeError("pipeline must have encode_prompt method")

    if not hasattr(pipeline, 'tokenizer'):
        raise TypeError("pipeline must have tokenizer attribute")

    try:
        # Handle empty prompts
        if not prompt or not prompt.strip():
            logger.debug("Encoding empty prompt")
            return pipeline.encode_prompt("")[0]

        # Tokenize the prompt
        tokens = pipeline.tokenizer.encode(prompt, add_special_tokens=False)

        # If prompt fits in standard limit, use normal encoding
        if len(tokens) <= 77:
            logger.debug("Prompt fits standard limit (%d tokens), using normal encoding", len(tokens))
            return pipeline.encode_prompt(prompt)[0]

        logger.info("Prompt exceeds token limit (%d tokens), using chunking approach", len(tokens))

        # Split into chunks
        chunks = _split_tokens_into_chunks(tokens, max_tokens)

        # Handle case where chunking results in too many chunks
        if len(chunks) > 10:  # Arbitrary limit to prevent memory issues
            logger.warning("Prompt resulted in %d chunks, truncating to first 10", len(chunks))
            chunks = chunks[:10]

        logger.debug("Processing %d chunks for long prompt", len(chunks))

        # Encode each chunk
        chunk_embeddings = []
        for i, chunk_tokens in enumerate(chunks):
            try:
                chunk_text = pipeline.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunk_embedding = pipeline.encode_prompt(chunk_text)[0]
                chunk_embeddings.append(chunk_embedding)
                logger.debug("Successfully encoded chunk %d/%d", i + 1, len(chunks))
            except (ValueError, RuntimeError, IndexError) as e:
                logger.error("Failed to encode chunk %d: %s", i, e)
                # Skip this chunk and continue
                continue

        if not chunk_embeddings:
            raise ValueError("Failed to encode any chunks of the prompt")

        # Concatenate all chunk embeddings
        combined_embeddings = torch.cat(chunk_embeddings, dim=1)

        logger.info("Successfully encoded prompt into %d chunks, final embedding shape: %s",
                    len(chunk_embeddings), combined_embeddings.shape)

        return combined_embeddings

    except (ValueError, RuntimeError, IndexError) as e:
        logger.error("Failed to encode long prompt: %s", e)
        # Fallback to standard encoding with truncation
        logger.info("Falling back to standard encoding with truncation")
        return pipeline.encode_prompt(prompt)[0]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected error encoding long prompt: %s", e)
        # Fallback to standard encoding with truncation
        logger.info("Falling back to standard encoding with truncation")
        return pipeline.encode_prompt(prompt)[0]


def load_loras(pipeline: DiffusionPipeline, loras: list, token: str) -> None:
    """ Load LoRA models into the pipeline """
    if not loras:
        return

    logger.info("Loading %d LoRA(s)", len(loras))
    for lora_config in loras:
        # Handle LoRA loading based on whether repo is specified
        if lora_config.repo:
            # Load LoRA file from within a repository
            logger.info("Loading LoRA: %s/%s (weight: %.2f)", lora_config.repo, lora_config.lora, lora_config.weight)
            pipeline.load_lora_weights(
                lora_config.repo,
                weight_name=lora_config.lora,
                token=token
            )
        else:
            # Load LoRA from a dedicated repository
            logger.info("Loading LoRA: %s (weight: %.2f)", lora_config.lora, lora_config.weight)
            pipeline.load_lora_weights(
                lora_config.lora,
                token=token
            )

        pipeline.fuse_lora(lora_scale=lora_config.weight)
