#!/usr/bin/env python3
"""
Test a distilled DreamingReasoningLLM model on a sample prompt.

Usage:
    python scripts/test_distilled_model.py --model ./models/distilled_from_oa --prompt "What is the meaning of life?"
"""


import argparse
import torch
from src.image_token_llm.dreaming_model import DreamingReasoningLLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Test a distilled model."
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model directory'
    )
    parser.add_argument(
        '--prompt', type=str,
        default="What is the meaning of life?",
        help='Prompt to test'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cpu or cuda)'
    )
    args = parser.parse_args()



    # Auto-select device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {args.model} on {args.device}...")
    try:
        model = DreamingReasoningLLM.load_pretrained(
            args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model.eval()

    # Load the same tokenizer as the model (default to bert-base-uncased)
    tokenizer_name = getattr(model.config, "hf_tokenizer_name",
                            "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"\nPrompt: {args.prompt}")
    with torch.no_grad():
        output = model.generate(
            prompt=args.prompt,
            max_length=64,
            temperature=0.8
        )

    # If output is not string, decode with tokenizer
    if isinstance(output, str):
        decoded = output
    elif isinstance(output, (list, tuple, torch.Tensor)):
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        decoded = tokenizer.decode(output, skip_special_tokens=True)
    else:
        decoded = str(output)

    print("\nModel output:")
    print(decoded)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test distilled DreamingReasoningLLM model")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--prompt", type=str, default="What is the meaning of life?", help="Prompt to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = DreamingReasoningLLM.load_pretrained(args.model, device=args.device)
    model.eval()

    print(f"Prompt: {args.prompt}")
    output = model.generate(prompt=args.prompt, max_length=64, temperature=0.8)
    print(f"Model output:\n{output}")
