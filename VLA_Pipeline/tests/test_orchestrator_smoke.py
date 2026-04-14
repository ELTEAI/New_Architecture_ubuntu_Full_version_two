from pathlib import Path

from src.pipeline.orchestrator import load_cfg


def test_load_cfg_smoke():
    cfg = load_cfg(Path("/home/ubuntu/New_Architecture/VLA_Pipeline/config/pipeline.yaml"))
    assert "pipeline" in cfg
    assert "execution" in cfg
    assert cfg["pipeline"]["mode"] in {"planner_only", "reflex_only", "hybrid"}

