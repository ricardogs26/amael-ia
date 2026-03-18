
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
import main
from main import app, PodStatus, ClusterSnapshot

client = TestClient(app)

def test_get_health_stats_all_running():
    pods = [
        PodStatus(name="pod-1", namespace="amael-ia", phase="Running", restart_count=0, 
                  waiting_reason="", last_state_reason="", owner_name="dep-1", 
                  owner_kind="Deployment", start_time=datetime.now(timezone.utc)),
        PodStatus(name="pod-2", namespace="vault", phase="Running", restart_count=0, 
                  waiting_reason="", last_state_reason="", owner_name="dep-2", 
                  owner_kind="Deployment", start_time=datetime.now(timezone.utc))
    ]
    snapshot = ClusterSnapshot(timestamp=datetime.now(timezone.utc), pods=pods, nodes=[])
    
    with patch("main.observe_cluster", return_value=snapshot):
        response = client.get("/api/sre/health-stats")
        assert response.status_code == 200
        data = response.json()
        assert data["failing_count"] == 0
        assert data["total_pods"] == 2

def test_get_health_stats_with_failing_pods():
    pods = [
        PodStatus(name="pod-running", namespace="amael-ia", phase="Running", restart_count=0, 
                  waiting_reason="", last_state_reason="", owner_name="dep-1", 
                  owner_kind="Deployment", start_time=datetime.now(timezone.utc)),
        PodStatus(name="pod-failed", namespace="vault", phase="Failed", restart_count=0, 
                  waiting_reason="", last_state_reason="", owner_name="dep-2", 
                  owner_kind="Deployment", start_time=datetime.now(timezone.utc)),
        PodStatus(name="pod-restarting", namespace="kong", phase="Running", restart_count=15, 
                  waiting_reason="", last_state_reason="", owner_name="dep-3", 
                  owner_kind="Deployment", start_time=datetime.now(timezone.utc))
    ]
    snapshot = ClusterSnapshot(timestamp=datetime.now(timezone.utc), pods=pods, nodes=[])
    
    with patch("main.observe_cluster", return_value=snapshot):
        response = client.get("/api/sre/health-stats")
        assert response.status_code == 200
        data = response.json()
        # failing count should be 2: one failed, one with >10 restarts
        assert data["failing_count"] == 2
        assert any("pod-failed" in p for p in data["failing_pods"])
        assert any("pod-restarting" in p for p in data["failing_pods"])
