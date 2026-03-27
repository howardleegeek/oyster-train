---
task_id: S14-deploy-integration
project: oyster-train
priority: 2
estimated_minutes: 45
depends_on: [S01, S08]
modifies: ["deploy/orchestrator.py", "deploy/docker-compose.yml"]
executor: glm
---
## Goal
Create deployment orchestration that ties together the Flower server, monitoring, and model distribution into a deployable stack.

## Constraints
- Container runtime: Docker Compose v2
- Services:
  1. `flower-server` - Python container running server/main.py with DiLoCo strategy
  2. `model-registry` - FastAPI service (deploy/api.py) serving model checkpoints
  3. `monitor` - Prometheus + Grafana for FL metrics
  4. `redis` - For client session management and rate limiting
- Flower server config via environment variables (from server/config.py ServerConfig)
- Model checkpoints saved after each FL round to shared volume
- Health checks on all services
- Phone clients connect via gRPC (Flower) port 8080

## Deliverables
- `deploy/docker-compose.yml` - Full stack definition with:
  - flower-server (port 8080 gRPC)
  - model-registry (port 8000 HTTP)
  - prometheus (port 9090)
  - grafana (port 3000)
  - redis (port 6379)
- `deploy/Dockerfile.server` - Flower server container
- `deploy/Dockerfile.registry` - Model registry container
- `deploy/orchestrator.py` - Deployment management script:
  - `deploy up` - Start all services
  - `deploy down` - Stop all services
  - `deploy status` - Health check all services
  - `deploy checkpoint` - Export latest model checkpoint
- `deploy/prometheus.yml` - Prometheus scrape config for FL metrics
- `deploy/grafana/dashboard.json` - Pre-built FL monitoring dashboard
- `tests/test_deploy.py` - Verify compose file is valid, Dockerfiles parse correctly

## Do NOT
- Modify server/ or compressor/ modules
- Include real credentials
- Actually run Docker (just create the files)
