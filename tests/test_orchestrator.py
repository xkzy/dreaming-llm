from image_token_llm import ReasoningOrchestrator, ExperimentConfig


def test_orchestrator_produces_result():
    orchestrator = ReasoningOrchestrator(ExperimentConfig())
    orchestrator.prime_graph(
        [
            ("cat", "chases", "mouse"),
            ("mouse", "hides", "hole"),
            ("cat", "waits", "hole"),
        ]
    )

    result = orchestrator.infer(["cat"])

    assert "best" in result
    assert result["score"] >= 0.0
