import torch
import pytest
from image_token_llm.config import ImageTokenizerConfig
from image_token_llm.image_tokenizer import ImageTokenizer

def test_image_tokenizer_clip():
    config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
    tokenizer = ImageTokenizer(config, backbone="clip")
    images = [torch.randn(3, 224, 224) for _ in range(2)]
    tokens = tokenizer.tokenize(images)
    assert len(tokens) == 2
    assert tokens[0].shape == (512,)
    assert tokens[1].shape == (512,)
    print("✓ ImageTokenizer with CLIP backbone produces correct output shape.")
