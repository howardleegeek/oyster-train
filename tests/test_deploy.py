#!/usr/bin/env python3
"""
Tests for OysterTrain deployment configuration
"""

import os
import yaml
import json
from pathlib import Path

def test_docker_compose_exists():
    """Test that docker-compose.yml exists and is valid YAML."""
    compose_path = Path("deploy/docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml not found"
    
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    
    assert data is not None, "docker-compose.yml is empty or invalid"
    assert "services" in data, "docker-compose.yml missing services section"
    
    # Check for required services
    services = data["services"]
    required_services = ["flower-server", "model-registry", "prometheus", "grafana", "redis"]
    for service in required_services:
        assert service in services, f"Missing required service: {service}"

def test_dockerfile_server_exists():
    """Test that Dockerfile.server exists."""
    dockerfile_path = Path("deploy/Dockerfile.server")
    assert dockerfile_path.exists(), "Dockerfile.server not found"
    
    with open(dockerfile_path) as f:
        content = f.read()
    
    assert "FROM python:" in content, "Dockerfile.server should use Python base image"
    assert "EXPOSE 8080" in content, "Dockerfile.server should expose port 8080"

def test_dockerfile_registry_exists():
    """Test that Dockerfile.registry exists."""
    dockerfile_path = Path("deploy/Dockerfile.registry")
    assert dockerfile_path.exists(), "Dockerfile.registry not found"
    
    with open(dockerfile_path) as f:
        content = f.read()
    
    assert "FROM python:" in content, "Dockerfile.registry should use Python base image"
    assert "EXPOSE 8000" in content, "Dockerfile.registry should expose port 8000"

def test_prometheus_config_exists():
    """Test that prometheus.yml exists and is valid YAML."""
    prometheus_path = Path("deploy/prometheus.yml")
    assert prometheus_path.exists(), "prometheus.yml not found"
    
    with open(prometheus_path) as f:
        data = yaml.safe_load(f)
    
    assert data is not None, "prometheus.yml is empty or invalid"
    assert "global" in data, "prometheus.yml missing global section"
    assert "scrape_configs" in data, "prometheus.yml missing scrape_configs section"

def test_grafana_dashboard_exists():
    """Test that grafana dashboard exists and is valid JSON."""
    dashboard_path = Path("deploy/grafana/dashboard.json")
    assert dashboard_path.exists(), "grafana/dashboard.json not found"
    
    with open(dashboard_path) as f:
        data = json.load(f)
    
    assert data is not None, "grafana/dashboard.json is empty or invalid"
    assert "title" in data, "grafana/dashboard.json missing title"
    assert "panels" in data, "grafana/dashboard.json missing panels"
    assert isinstance(data["panels"], list), "grafana/dashboard.json panels should be a list"

def test_orchestrator_exists():
    """Test that orchestrator.py exists."""
    orchestrator_path = Path("deploy/orchestrator.py")
    assert orchestrator_path.exists(), "orchestrator.py not found"
    
    # Check that it's executable
    assert os.access(orchestrator_path, os.X_OK), "orchestrator.py should be executable"

if __name__ == "__main__":
    test_docker_compose_exists()
    test_dockerfile_server_exists()
    test_dockerfile_registry_exists()
    test_prometheus_config_exists()
    test_grafana_dashboard_exists()
    test_orchestrator_exists()
    print("All deployment tests passed!")