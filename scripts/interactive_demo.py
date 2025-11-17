#!/usr/bin/env python3
"""Interactive demo for the Image-Token Reasoning LLM.

This script provides an interactive REPL for testing the pretrained model
with various prompts and configurations.

Usage:
    python scripts/interactive_demo.py ./pretrained_llama3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image_token_llm.model import ImageTokenReasoningLLM


def print_banner() -> None:
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print(" " * 15 + "Image-Token Reasoning LLM Demo")
    print("=" * 70)
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 70 + "\n")


def print_help() -> None:
    """Print help message."""
    print("\nAvailable commands:")
    print("  help              - Show this help message")
    print("  quit / exit       - Exit the demo")
    print("  temp <float>      - Set temperature (default: 0.8)")
    print("  tokens <int>      - Set max tokens (default: 50)")
    print("  stream on/off     - Enable/disable streaming")
    print("  info              - Show model information")
    print("  <your prompt>     - Generate response")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive demo for pretrained model"
    )
    parser.add_argument(
        "bundle_dir",
        type=str,
        help="Path to pretrained model bundle directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Verify bundle exists
    bundle_path = Path(args.bundle_dir)
    if not bundle_path.exists():
        print(f"Error: Bundle directory not found: {args.bundle_dir}")
        sys.exit(1)

    # Load model
    print("Loading model from bundle...")
    try:
        model = ImageTokenReasoningLLM.load_from_bundle(
            bundle_dir=args.bundle_dir,
            device=args.device,
            enable_rl=False,
        )
        print(f"✓ Model loaded from {args.bundle_dir}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # Configuration
    temperature = 0.8
    max_tokens = 50
    streaming = False

    print_banner()

    # Interactive loop
    while True:
        try:
            prompt = input("\n> ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            elif prompt.lower() == "help":
                print_help()
                continue

            elif prompt.lower().startswith("temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print("Invalid temperature value")
                continue

            elif prompt.lower().startswith("tokens "):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except (ValueError, IndexError):
                    print("Invalid token count")
                continue

            elif prompt.lower().startswith("stream "):
                try:
                    value = prompt.split()[1].lower()
                    streaming = value in ["on", "true", "1"]
                    print(
                        f"Streaming {'enabled' if streaming else 'disabled'}"
                    )
                except IndexError:
                    print("Usage: stream on/off")
                continue

            elif prompt.lower() == "info":
                print("\nModel Information:")
                print(f"  Device: {model.device}")
                print(f"  RL enabled: {model.enable_rl}")
                print(
                    f"  Vocab size: {model.config.text_decoder.vocab_size}"
                )
                print(
                    f"  Embed dim: "
                    f"{model.config.image_tokenizer.embedding_dim}"
                )
                print(f"  Graph nodes: {len(model.graph_rag.graph.nodes)}")
                if model.last_metadata:
                    print("\n  Last generation metadata:")
                    for key, value in model.last_metadata.items():
                        print(f"    {key}: {value}")
                continue

            # Generate response
            print("\nGenerating response...")
            print("-" * 70)

            try:
                if streaming:
                    result = model.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    print("Output: ", end="", flush=True)
                    for chunk in result:
                        print(chunk, end="", flush=True)
                    print()
                else:
                    output = model.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                    )
                    print(f"Output: {output}")

                print("-" * 70)

                # Show metadata
                if model.last_metadata:
                    print(
                        f"Seeds: {model.last_metadata.get('seeds', [])}"
                    )
                    if model.last_metadata.get("rl_metrics"):
                        print(
                            f"RL metrics: "
                            f"{model.last_metadata['rl_metrics']}"
                        )

            except Exception as e:
                print(f"Error during generation: {e}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
