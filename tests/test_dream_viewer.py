"""Tests for dream viewing functionality."""

import torch

from image_token_llm.config import DreamingConfig, ExperimentConfig
from image_token_llm.dream_viewer import DreamViewer, LiveDreamStream
from image_token_llm.dreaming_model import DreamingReasoningLLM


def test_dream_viewer_capture():
    """Test capturing dream states."""
    viewer = DreamViewer(enable_recording=True)
    
    # Capture some dream states
    dream_state = torch.randn(1, 10, 256)
    
    snapshot1 = viewer.capture_dream_state(
        step=0,
        dream_state=dream_state,
        metadata={"prompt": "test"}
    )
    
    assert snapshot1["step"] == 0
    assert "dream_mean" in snapshot1
    assert "dream_std" in snapshot1
    assert "metadata" in snapshot1
    assert len(viewer.recorded_dreams) == 1


def test_dream_viewer_formatting():
    """Test dream state formatting."""
    viewer = DreamViewer(enable_recording=True)
    
    dream_state = torch.randn(2, 10, 256)
    graph_state = {
        "num_nodes": 20,
        "num_edges": 40,
        "active_nodes": [0, 1, 2, 3, 4, 5]
    }
    
    snapshot = viewer.capture_dream_state(
        step=1,
        dream_state=dream_state,
        graph_state=graph_state
    )
    
    formatted = viewer.format_dream_state(snapshot, verbose=True)
    
    assert "Dream Step 1" in formatted
    assert "Shape:" in formatted
    assert "Mean:" in formatted
    assert "Graph:" in formatted
    assert "20 nodes" in formatted


def test_dream_viewer_summary():
    """Test dream summary statistics."""
    viewer = DreamViewer(enable_recording=True)
    
    # Record multiple dreams
    for i in range(5):
        dream_state = torch.randn(1, 10, 256)
        viewer.capture_dream_state(step=i, dream_state=dream_state)
    
    summary = viewer.get_dream_summary()
    
    assert summary["total_dreams"] == 5
    assert summary["total_steps"] == 5
    assert "avg_mean" in summary
    assert "avg_std" in summary
    assert "mean_range" in summary


def test_dream_viewer_export(tmp_path):
    """Test exporting dreams to file."""
    viewer = DreamViewer(enable_recording=True)
    
    # Record some dreams
    for i in range(3):
        dream_state = torch.randn(1, 10, 256)
        viewer.capture_dream_state(step=i, dream_state=dream_state)
    
    # Export as JSON
    json_path = tmp_path / "dreams.json"
    viewer.export_dreams(str(json_path), format="json")
    assert json_path.exists()
    
    # Export as text
    txt_path = tmp_path / "dreams.txt"
    viewer.export_dreams(str(txt_path), format="txt")
    assert txt_path.exists()


def test_live_dream_stream():
    """Test live dream streaming."""
    viewer = DreamViewer(enable_recording=True)
    stream = LiveDreamStream(viewer)
    
    # Test context manager
    with stream:
        assert stream.active
        dream_state = torch.randn(1, 10, 256)
        stream.stream_dream(dream_state=dream_state, step=0)
    
    assert not stream.active
    assert len(viewer.recorded_dreams) == 1


def test_model_with_dream_viewer():
    """Test DreamingReasoningLLM with dream viewer enabled."""
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,
        dream_length=3,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device="cpu",
        embedding_dim=128,
        vocab_size=256,
        enable_rl=False,
        enable_dream_viewer=True  # Enable viewer!
    )
    
    # Check viewer is initialized
    assert model.dream_viewer is not None
    assert model.enable_dream_viewer
    
    # Get viewer
    viewer = model.get_dream_viewer()
    assert viewer is not None
    assert isinstance(viewer, DreamViewer)


def test_model_generate_with_dream_watching(capsys):
    """Test generation with dream watching."""
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,
        dream_length=2,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device="cpu",
        embedding_dim=128,
        vocab_size=256,
        enable_rl=False,
        enable_dream_viewer=True
    )
    
    # Generate with dream watching
    result = model.generate(
        prompt="test",
        max_length=10,
        return_dreams=True,
        watch_dreams=True  # Enable live streaming!
    )
    
    # Check output
    captured = capsys.readouterr()
    assert "LIVE DREAM STREAM" in captured.out
    assert "Dream Step" in captured.out
    
    # Check dreams were recorded
    viewer = model.get_dream_viewer()
    assert len(viewer.recorded_dreams) > 0


def test_model_export_dreams(tmp_path):
    """Test exporting dreams from model."""
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,
        dream_length=2,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device="cpu",
        embedding_dim=128,
        vocab_size=256,
        enable_rl=False,
        enable_dream_viewer=True
    )
    
    # Generate to record dreams
    model.generate(
        prompt="test",
        max_length=10,
        watch_dreams=True
    )
    
    # Export dreams
    output_path = tmp_path / "model_dreams.json"
    model.export_recorded_dreams(str(output_path), format="json")
    assert output_path.exists()


def test_model_without_dream_viewer():
    """Test that model works without dream viewer."""
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,
        dream_length=2,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device="cpu",
        embedding_dim=128,
        vocab_size=256,
        enable_rl=False,
        enable_dream_viewer=False  # Disabled
    )
    
    # Check viewer is None
    assert model.dream_viewer is None
    assert not model.enable_dream_viewer
    
    # Generate should work without viewer
    result = model.generate(
        prompt="test",
        max_length=10,
        watch_dreams=False  # Won't watch since viewer disabled
    )
    
    assert result is not None


def test_dream_viewer_clear():
    """Test clearing recorded dreams."""
    viewer = DreamViewer(enable_recording=True)
    
    # Record some dreams
    for i in range(3):
        dream_state = torch.randn(1, 10, 256)
        viewer.capture_dream_state(step=i, dream_state=dream_state)
    
    assert len(viewer.recorded_dreams) == 3
    
    # Clear
    viewer.clear_recording()
    
    assert len(viewer.recorded_dreams) == 0
    assert viewer.current_step == 0
