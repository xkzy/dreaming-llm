#!/usr/bin/env python3
"""
Simple CLI for Image-Token LLM - Easy model interaction.

Usage:
    python scripts/llm_chat.py --model ./pretrained_llama3/
    python scripts/llm_chat.py --model ./pretrained_enhanced/ --temp 0.9
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_token_llm.model import ImageTokenReasoningLLM


def chat_loop(
    model: ImageTokenReasoningLLM,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> None:
    """Interactive chat loop."""
    
    print("\n" + "=" * 60)
    print("Image-Token LLM - Interactive Chat")
    print("=" * 60)
    print("Type your prompt and press Enter.")
    print("Commands:")
    print("  /temp <value>  - Set temperature (0.1-2.0)")
    print("  /tokens <num>  - Set max tokens (10-500)")
    print("  /stream on|off - Toggle streaming")
    print("  /help          - Show this help")
    print("  /quit or /exit - Exit chat")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.lower().split(maxsplit=1)
                cmd = cmd_parts[0]
                
                if cmd in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break
                
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /temp <value>  - Set temperature")
                    print("  /tokens <num>  - Set max tokens")
                    print("  /stream on|off - Toggle streaming")
                    print("  /help          - Show this help")
                    print("  /quit or /exit - Exit\n")
                    continue
                
                elif cmd == "/temp":
                    if len(cmd_parts) < 2:
                        print("Usage: /temp <value>")
                        continue
                    try:
                        new_temp = float(cmd_parts[1])
                        if 0.0 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"Temperature set to {temperature}")
                        else:
                            print("Temperature must be between 0.0 and 2.0")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                
                elif cmd == "/tokens":
                    if len(cmd_parts) < 2:
                        print("Usage: /tokens <number>")
                        continue
                    try:
                        new_tokens = int(cmd_parts[1])
                        if 10 <= new_tokens <= 500:
                            max_tokens = new_tokens
                            print(f"Max tokens set to {max_tokens}")
                        else:
                            print("Tokens must be between 10 and 500")
                    except ValueError:
                        print("Invalid token value")
                    continue
                
                elif cmd == "/stream":
                    if len(cmd_parts) < 2:
                        print("Usage: /stream on|off")
                        continue
                    toggle = cmd_parts[1].lower()
                    if toggle == "on":
                        stream = True
                        print("Streaming enabled")
                    elif toggle == "off":
                        stream = False
                        print("Streaming disabled")
                    else:
                        print("Use: /stream on or /stream off")
                    continue
                
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")
                    continue
            
            # Generate response
            print("AI: ", end="", flush=True)
            
            result = model.generate(
                prompt=user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            
            if stream:
                for chunk in result:
                    print(chunk, end="", flush=True)
                print()  # newline
            else:
                print(result)
            
            print()  # blank line after response
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit or continue chatting.")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue


def single_prompt(
    model: ImageTokenReasoningLLM,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> None:
    """Generate response for single prompt."""
    
    result = model.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    
    if stream:
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result)


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI for Image-Token LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model bundle directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt (skips chat mode)",
    )
    parser.add_argument(
        "--temp",
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (0.1-2.0, default: 0.7)",
    )
    parser.add_argument(
        "--tokens",
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (10-500, default: 100)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model bundle not found: {args.model}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {args.model}...")
    try:
        model = ImageTokenReasoningLLM.load_from_bundle(
            bundle_dir=args.model,
            device=args.device,
            enable_rl=False,
        )
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Single prompt or chat mode
    if args.prompt:
        single_prompt(
            model=model,
            prompt=args.prompt,
            temperature=args.temp,
            max_tokens=args.tokens,
            stream=args.stream,
        )
    else:
        chat_loop(
            model=model,
            temperature=args.temp,
            max_tokens=args.tokens,
            stream=args.stream,
        )


if __name__ == "__main__":
    main()
