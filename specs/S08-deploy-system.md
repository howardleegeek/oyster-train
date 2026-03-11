---
task_id: S08-deploy-system
project: oyster-train
priority: 2
estimated_minutes: 45
depends_on: []
modifies: ["deploy/"]
executor: glm
---

## Goal
Build the deployment and phone management system: device registration, model distribution (OTA), training scheduling, and monitoring dashboard backend.

## Context
- 10K UBS1 phones need to register with the training server
- Model updates distributed via OTA (phones download new global model periodically)
- Training schedule: only when charging + WiFi + idle
- Need monitoring: how many phones active, convergence progress, bandwidth usage

## Deliverables

### deploy/registration_server.py
- FastAPI server for phone registration and management
- Endpoints:
  - `POST /api/register`: Phone registers with device_id, hardware_info, os_version
  - `GET /api/config/{device_id}`: Return training config for this device
  - `GET /api/model/latest`: Return URL to download latest global model
  - `POST /api/heartbeat`: Phone sends status (battery, wifi, training_active, steps_done)
  - `GET /api/stats`: Return fleet statistics
- SQLite database for device registry (simple, no external deps)
- Rate limiting: max 1 registration per device per day

### deploy/model_distributor.py
- Manage global model versions and distribution
- `ModelDistributor` class:
  - `publish_model(model_path, version) -> str`: Upload new model, return download URL
  - `get_latest_version() -> ModelVersion`: Current global model info
  - `get_download_url(version) -> str`: Signed URL for model download
  - `cleanup_old_versions(keep=3)`: Remove old model files
- Store models locally in deploy/models/ directory
- Serve via FastAPI static files (or S3 URL generation)

### deploy/scheduler.py
- Training schedule management
- `TrainingScheduler` class:
  - `should_train(device_status) -> bool`:
    - Check: is_charging AND is_wifi AND battery > 30%
    - Check: not in quiet hours (configurable, default 11pm-7am)
    - Check: daily training quota not exceeded (max 4 hours/day)
  - `get_training_window(device_id) -> TimeWindow`:
    - Stagger training windows to avoid all phones syncing simultaneously
    - Use device_id hash to assign time slots
  - `update_schedule(new_config)`: Update global schedule config

### deploy/monitor.py
- Fleet monitoring and metrics collection
- `FleetMonitor` class:
  - `record_heartbeat(device_id, status)`: Store device status
  - `get_active_devices() -> int`: Count devices currently training
  - `get_fleet_stats() -> FleetStats`:
    - total_registered, active_now, trained_today
    - avg_battery_level, wifi_connected_count
    - total_steps_today, total_bytes_uploaded
    - convergence_curve (loss per round)
  - `get_device_history(device_id) -> List[HeartbeatRecord]`
- Store in SQLite (same DB as registration)

### deploy/dashboard_api.py
- FastAPI endpoints for monitoring dashboard
- Endpoints:
  - `GET /dashboard/overview`: Fleet summary stats
  - `GET /dashboard/devices`: List all devices with status
  - `GET /dashboard/training-curve`: Loss/accuracy per round
  - `GET /dashboard/bandwidth`: Total bandwidth usage
  - `GET /dashboard/device/{device_id}`: Single device detail
- CORS enabled for frontend access
- JSON response format

### deploy/requirements.txt
- fastapi>=0.100
- uvicorn
- pydantic>=2.0
- aiosqlite (async SQLite for FastAPI)
- python-multipart

### deploy/Dockerfile
- Simple Dockerfile for deployment server
- Python 3.11 slim base
- Install requirements
- Expose ports 8000 (API) + 8080 (Flower, if co-located)
- CMD: uvicorn deploy.registration_server:app

### deploy/docker-compose.yml
- Services: registration_server, flower_server (from S01)
- Volumes: models/, data/
- Network: oyster-train-net
- Health checks

## Constraints
- Python 3.10+, FastAPI
- SQLite for simplicity (can migrate to PostgreSQL later)
- No authentication for PoC (add in Phase 2)
- All endpoints return JSON
- Write tests in tests/test_deploy.py
- No frontend UI code (API only, dashboard frontend is separate project)

## Acceptance Criteria
- [ ] `uvicorn deploy.registration_server:app` starts on port 8000
- [ ] Registration endpoint creates device record
- [ ] Heartbeat endpoint updates device status
- [ ] Stats endpoint returns correct fleet metrics
- [ ] Scheduler correctly determines training windows
- [ ] Model distributor serves model files
- [ ] pytest tests/test_deploy.py passes
- [ ] Dockerfile builds successfully
