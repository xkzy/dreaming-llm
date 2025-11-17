# Architecture Overview

## Token Thinking via Image Triplets
- Inputs are normalized into `(what, action, result)` image triplets.
- `ImageTokenizer` slices each tensor into patch tokens and produces signatures used as graph nodes.

## Graph RAG Fabric
- Triplet signatures seed a multi-digraph storing typed relationships.
- Retrieval expands via k-nearest neighbor traversals within configurable hop limits.

## Multi-Situation Simulation
- `MultiScenarioSimulator` launches branch rollouts across retrieved neighborhoods.
- Paths retain provenance for downstream arbitration.

## Evaluation Loop
- `ScenarioEvaluator` applies tunable heuristics to score reasoning chains.
- The orchestrator surfaces the highest-confidence path plus GPT-5.1 preview flag status for clients.
