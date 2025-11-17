"""Dream Viewer - Peek into what the model is dreaming.

This module provides visualization and inspection tools to see the
model's internal dream states during generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class DreamViewer:
    """Visualize and inspect dream sequences from DreamingReasoningLLM."""
    
    def __init__(self, enable_recording: bool = True):
        """Initialize dream viewer.
        
        Args:
            enable_recording: Whether to record dreams for playback
        """
        self.enable_recording = enable_recording
        self.recorded_dreams: List[Dict[str, Any]] = []
        self.current_step = 0
        
    def capture_dream_state(
        self,
        step: int,
        dream_state: torch.Tensor,
        graph_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Capture a single dream state snapshot.
        
        Args:
            step: Current dream step number
            dream_state: Dream state tensor (B, seq_len, dim)
            graph_state: Optional graph reasoning state
            metadata: Optional additional metadata
            
        Returns:
            Captured dream snapshot
        """
        snapshot = {
            "step": step,
            "timestamp": self.current_step,
            "dream_shape": list(dream_state.shape),
            "dream_mean": float(dream_state.mean().item()),
            "dream_std": float(dream_state.std().item()),
            "dream_min": float(dream_state.min().item()),
            "dream_max": float(dream_state.max().item()),
        }
        
        # Add graph state if available
        if graph_state is not None:
            snapshot["graph"] = {
                "num_nodes": graph_state.get("num_nodes", 0),
                "num_edges": graph_state.get("num_edges", 0),
                "active_nodes": graph_state.get("active_nodes", []),
            }
        
        # Add metadata
        if metadata is not None:
            snapshot["metadata"] = metadata
            
        # Record if enabled
        if self.enable_recording:
            self.recorded_dreams.append(snapshot)
            
        self.current_step += 1
        return snapshot
    
    def format_dream_state(
        self,
        snapshot: Dict[str, Any],
        verbose: bool = True
    ) -> str:
        """Format a dream snapshot as human-readable text.
        
        Args:
            snapshot: Dream snapshot dictionary
            verbose: Include detailed statistics
            
        Returns:
            Formatted dream state string
        """
        lines = []
        lines.append(f"🌙 Dream Step {snapshot['step']}")
        lines.append("-" * 50)
        
        # Dream tensor info
        shape = snapshot["dream_shape"]
        lines.append(f"  Shape: {shape}")
        lines.append(f"  Mean:  {snapshot['dream_mean']:.4f}")
        lines.append(f"  Std:   {snapshot['dream_std']:.4f}")
        
        if verbose:
            lines.append(f"  Min:   {snapshot['dream_min']:.4f}")
            lines.append(f"  Max:   {snapshot['dream_max']:.4f}")
        
        # Graph info
        if "graph" in snapshot:
            graph = snapshot["graph"]
            lines.append(
                f"  Graph: {graph['num_nodes']} nodes, "
                f"{graph['num_edges']} edges"
            )
            if graph.get("active_nodes"):
                active = ", ".join(map(str, graph["active_nodes"][:5]))
                lines.append(f"  Active: [{active}...]")
        
        # Metadata
        if "metadata" in snapshot:
            meta = snapshot["metadata"]
            if meta:
                lines.append("  Meta:")
                for key, val in meta.items():
                    lines.append(f"    {key}: {val}")
        
        return "\n".join(lines)
    
    def watch_dreams(
        self,
        dreams: List[Dict[str, Any]],
        interval: float = 0.5,
        live: bool = False
    ) -> None:
        """Watch dream sequences unfold.
        
        Args:
            dreams: List of dream snapshots
            interval: Time between frames (seconds)
            live: Whether to print in real-time or batch
        """
        import time
        
        print("\n" + "=" * 60)
        print("🔮 DREAM VIEWER - WATCHING MODEL DREAMS")
        print("=" * 60 + "\n")
        
        for i, dream in enumerate(dreams, 1):
            if live and i > 1:
                time.sleep(interval)
                
            print(self.format_dream_state(dream))
            print()
        
        print("=" * 60)
        print(f"✨ Viewed {len(dreams)} dream states")
        print("=" * 60 + "\n")
    
    def export_dreams(
        self,
        output_path: str,
        format: str = "json"
    ) -> None:
        """Export recorded dreams to file.
        
        Args:
            output_path: Path to save dreams
            format: Export format (json, txt)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(path, "w") as f:
                json.dump(self.recorded_dreams, f, indent=2)
                
        elif format == "txt":
            with open(path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("DREAM RECORDING\n")
                f.write("=" * 60 + "\n\n")
                
                for dream in self.recorded_dreams:
                    f.write(self.format_dream_state(dream))
                    f.write("\n\n")
                    
                f.write("=" * 60 + "\n")
                f.write(f"Total dreams: {len(self.recorded_dreams)}\n")
                f.write("=" * 60 + "\n")
        
        print(f"💾 Saved {len(self.recorded_dreams)} dreams to {path}")
    
    def clear_recording(self) -> None:
        """Clear recorded dreams."""
        self.recorded_dreams.clear()
        self.current_step = 0
    
    def get_dream_summary(self) -> Dict[str, Any]:
        """Get summary statistics of recorded dreams.
        
        Returns:
            Summary dictionary
        """
        if not self.recorded_dreams:
            return {
                "total_dreams": 0,
                "total_steps": 0
            }
        
        means = [d["dream_mean"] for d in self.recorded_dreams]
        stds = [d["dream_std"] for d in self.recorded_dreams]
        
        return {
            "total_dreams": len(self.recorded_dreams),
            "total_steps": self.current_step,
            "avg_mean": sum(means) / len(means),
            "avg_std": sum(stds) / len(stds),
            "mean_range": (min(means), max(means)),
            "std_range": (min(stds), max(stds)),
        }
    
    def create_dream_heatmap(
        self,
        dream_sequence: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Create a heatmap visualization of dream evolution.
        
        Args:
            dream_sequence: List of dream snapshots
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠️  matplotlib not available, skipping heatmap")
            return None
        
        if not dream_sequence:
            print("⚠️  No dreams to visualize")
            return None
        
        # Extract statistics
        steps = [d["step"] for d in dream_sequence]
        means = [d["dream_mean"] for d in dream_sequence]
        stds = [d["dream_std"] for d in dream_sequence]
        mins = [d["dream_min"] for d in dream_sequence]
        maxs = [d["dream_max"] for d in dream_sequence]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("🌙 Dream State Evolution", fontsize=16)
        
        # Mean plot
        axes[0, 0].plot(steps, means, 'b-', linewidth=2)
        axes[0, 0].set_title("Dream Mean")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Mean Value")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Std plot
        axes[0, 1].plot(steps, stds, 'r-', linewidth=2)
        axes[0, 1].set_title("Dream Std Dev")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Std Dev")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Min/Max plot
        axes[1, 0].plot(steps, mins, 'g-', label="Min", linewidth=2)
        axes[1, 0].plot(steps, maxs, 'm-', label="Max", linewidth=2)
        axes[1, 0].set_title("Dream Range")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined heatmap
        data = np.array([means, stds, mins, maxs])
        im = axes[1, 1].imshow(data, aspect='auto', cmap='viridis')
        axes[1, 1].set_title("Combined Heatmap")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_yticks([0, 1, 2, 3])
        axes[1, 1].set_yticklabels(["Mean", "Std", "Min", "Max"])
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved dream heatmap to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None


class LiveDreamStream:
    """Stream dreams in real-time during model generation."""
    
    def __init__(self, viewer: DreamViewer):
        """Initialize live dream stream.
        
        Args:
            viewer: DreamViewer instance
        """
        self.viewer = viewer
        self.active = False
        
    def __enter__(self):
        """Start streaming."""
        self.active = True
        print("\n" + "=" * 60)
        print("🔮 LIVE DREAM STREAM - ACTIVE")
        print("=" * 60 + "\n")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop streaming."""
        self.active = False
        print("\n" + "=" * 60)
        print("✨ STREAM ENDED")
        print("=" * 60 + "\n")
        
        # Show summary
        summary = self.viewer.get_dream_summary()
        print(f"Captured {summary['total_dreams']} dream states")
        if summary['total_dreams'] > 0:
            print(f"Avg Mean: {summary['avg_mean']:.4f}")
            print(f"Avg Std:  {summary['avg_std']:.4f}")
        print()
    
    def stream_dream(
        self,
        dream_state: torch.Tensor,
        step: int,
        graph_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stream a single dream state.
        
        Args:
            dream_state: Dream state tensor
            step: Current step
            graph_state: Optional graph state
            metadata: Optional metadata
        """
        if not self.active:
            return
        
        snapshot = self.viewer.capture_dream_state(
            step=step,
            dream_state=dream_state,
            graph_state=graph_state,
            metadata=metadata
        )
        
        # Print immediately for live viewing
        print(self.viewer.format_dream_state(snapshot, verbose=False))
        print()
