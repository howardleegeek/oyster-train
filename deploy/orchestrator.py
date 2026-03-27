#!/usr/bin/env python3
"""
Orchestrator for OysterTrain Federated Learning Deployment
Manages the docker-compose stack for Flower server, model registry, monitoring, and Redis.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class DeploymentOrchestrator:
    def __init__(self, compose_file: str = "deploy/docker-compose.yml"):
        self.compose_file = compose_file
        self.project_dir = Path(__file__).parent
        
    def run_command(self, cmd: List[str]) -> tuple[int, str, str]:
        """Run a command and return exit code, stdout, and stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def up(self) -> bool:
        """Start all services."""
        print("Starting OysterTrain deployment...")
        cmd = ["docker-compose", "-f", self.compose_file, "up", "-d"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code == 0:
            print("✓ All services started successfully")
            print(stdout)
            return True
        else:
            print("✗ Failed to start services")
            print(stderr)
            return False
    
    def down(self) -> bool:
        """Stop all services."""
        print("Stopping OysterTrain deployment...")
        cmd = ["docker-compose", "-f", self.compose_file, "down"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code == 0:
            print("✓ All services stopped successfully")
            print(stdout)
            return True
        else:
            print("✗ Failed to stop services")
            print(stderr)
            return False
    
    def status(self) -> bool:
        """Check status of all services."""
        print("Checking status of OysterTrain services...")
        cmd = ["docker-compose", "-f", self.compose_file, "ps"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code == 0:
            print("Service status:")
            print(stdout)
            
            # Also get detailed health info
            cmd = ["docker-compose", "-f", self.compose_file, "ps", "--format", "json"]
            exit_code, stdout, stderr = self.run_command(cmd)
            
            if exit_code == 0 and stdout.strip():
                try:
                    services = [json.loads(line) for line in stdout.strip().split('\n') if line.strip()]
                    print("\nDetailed health check:")
                    for service in services:
                        name = service.get('Name', 'unknown')
                        state = service.get('State', 'unknown')
                        health = service.get('Health', 'none')
                        print(f"  {name}: {state} (health: {health})")
                except:
                    pass  # Fall back to basic output
            
            return True
        else:
            print("✗ Failed to get service status")
            print(stderr)
            return False
    
    def checkpoint(self) -> bool:
        """Export latest model checkpoint."""
        print("Exporting latest model checkpoint...")
        
        # Check if checkpoints directory exists and has files
        checkpoints_dir = self.project_dir / "checkpoints"
        if not checkpoints_dir.exists():
            print("✗ Checkpoints directory does not exist")
            return False
            
        # List checkpoint files
        checkpoint_files = list(checkpoints_dir.glob("*.pt")) + list(checkpoints_dir.glob("*.ckpt"))
        
        if not checkpoint_files:
            print("✗ No checkpoint files found in checkpoints directory")
            return False
            
        # Get the latest checkpoint file
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        print(f"✓ Latest checkpoint: {latest_checkpoint.name}")
        print(f"  Size: {latest_checkpoint.stat().st_size / (1024*1024):.2f} MB")
        print(f"  Modified: {time.ctime(latest_checkpoint.stat().st_mtime)}")
        
        # In a real implementation, we might copy this to a distribution location
        # or generate a download URL, but for now we just report it
        return True
    
    def logs(self, service: Optional[str] = None) -> bool:
        """Show logs for services."""
        if service is not None:
            print(f"Showing logs for {service}...")
            cmd = ["docker-compose", "-f", self.compose_file, "logs", "--tail", "100", service]
        else:
            print("Showing logs for all services...")
            cmd = ["docker-compose", "-f", self.compose_file, "logs", "--tail", "100"]
            
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code == 0:
            print(stdout)
            return True
        else:
            print("✗ Failed to get logs")
            print(stderr)
            return False

def print_usage():
    """Print usage information."""
    print("""
OysterTrain Deployment Orchestrator

Usage:
  python orchestrator.py up           - Start all services
  python orchestrator.py down         - Stop all services
  python orchestrator.py status       - Check status of all services
  python orchestrator.py checkpoint   - Export latest model checkpoint
  python orchestrator.py logs [service] - Show logs for services (or specific service)

Services:
  flower-server   - Flower federated learning server (port 8080)
  model-registry  - Model registry and device management API (port 8000)
  prometheus      - Monitoring and alerting system (port 9090)
  grafana         - Visualization dashboard (port 3000)
  redis           - In-memory data store for caching and session management (port 6379)
    """)

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    orchestrator = DeploymentOrchestrator()
    
    if command == "up":
        success = orchestrator.up()
        sys.exit(0 if success else 1)
    elif command == "down":
        success = orchestrator.down()
        sys.exit(0 if success else 1)
    elif command == "status":
        success = orchestrator.status()
        sys.exit(0 if success else 1)
    elif command == "checkpoint":
        success = orchestrator.checkpoint()
        sys.exit(0 if success else 1)
    elif command == "logs":
        service = sys.argv[2] if len(sys.argv) > 2 else None
        success = orchestrator.logs(service)
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()