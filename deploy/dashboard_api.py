#!/usr/bin/env python3
"""
Dashboard API for OysterTrain Federated Learning System
Provides endpoints for monitoring dashboard frontend.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import asyncio
import aiosqlite
from datetime import datetime, timedelta

from .registration_server import app as registration_app, get_db
from .monitor import FleetMonitor
from .model_distributor import ModelDistributor

# Create FastAPI app for dashboard
app = FastAPI(title="OysterTrain Dashboard API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
monitor = FleetMonitor()
model_distributor = ModelDistributor()

# Dependency to get database connection
async def get_db():
    async with aiosqlite.connect("deploy/devices.db") as db:
        db.row_factory = aiosqlite.Row
        yield db

# Dashboard endpoints
@app.get("/dashboard/overview")
async def get_dashboard_overview():
    """Get fleet summary stats for dashboard overview."""
    stats = await monitor.get_fleet_stats()
    training_progress = await monitor.get_training_progress()
    
    return {
        "fleet_stats": stats.to_dict(),
        "training_progress": training_progress,
        "latest_model": model_distributor.get_latest_version().__dict__ if model_distributor.get_latest_version() else None
    }

@app.get("/dashboard/devices")
async def get_dashboard_devices(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    training_only: bool = False
):
    """List all devices with status for dashboard."""
    async with aiosqlite.connect("deploy/devices.db") as db:
        db.row_factory = aiosqlite.Row
        
        # Base query
        query = """
            SELECT d.device_id, d.hardware_info, d.os_version, d.registered_at, d.last_seen,
                   h.battery_level, h.wifi_connected, h.training_active, h.steps_done, h.timestamp
            FROM devices d
            LEFT JOIN (
                SELECT device_id, battery_level, wifi_connected, training_active, steps_done, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY timestamp DESC) as rn
                FROM heartbeats
            ) h ON d.device_id = h.device_id AND h.rn = 1
        """
        
        if training_only:
            query += " WHERE h.training_active = 1"
        
        query += " ORDER BY d.last_seen DESC LIMIT ? OFFSET ?"
        
        async with db.execute(query, (limit, offset)) as cursor:
            rows = await cursor.fetchall()
            
            devices = []
            for row in rows:
                device = {
                    "device_id": row[0],
                    "hardware_info": row[1],
                    "os_version": row[2],
                    "registered_at": row[3],
                    "last_seen": row[4],
                    "battery_level": row[5],
                    "wifi_connected": bool(row[6]) if row[6] is not None else False,
                    "training_active": bool(row[7]) if row[7] is not None else False,
                    "steps_done": row[8] if row[8] is not None else 0,
                    "last_heartbeat": row[9]
                }
                devices.append(device)
            
            return {"devices": devices, "count": len(devices)}

@app.get("/dashboard/training-curve")
async def get_training_curve():
    """Get loss/accuracy per round for training curve."""
    # In a real implementation, this would come from training metrics
    # For now, return placeholder data
    return {
        "rounds": list(range(1, 11)),
        "loss": [0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22, 0.21, 0.20],
        "accuracy": [0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84, 0.845, 0.85]
    }

@app.get("/dashboard/bandwidth")
async def get_bandwidth_usage():
    """Get total bandwidth usage statistics."""
    # Placeholder - would need actual bandwidth tracking
    return {
        "total_uploaded_mb": 1024.5,
        "total_downloaded_mb": 2048.0,
        "avg_per_device_kb": 150.2,
        "bandwidth_over_time": [
            {"hour": i, "uploaded_mb": i * 10.5, "downloaded_mb": i * 20.3}
            for i in range(24)
        ]
    }

@app.get("/dashboard/device/{device_id}")
async def get_device_detail(device_id: str):
    """Get single device detail."""
    async with aiosqlite.connect("deploy/devices.db") as db:
        db.row_factory = aiosqlite.Row
        
        # Get device info
        async with db.execute(
            "SELECT * FROM devices WHERE device_id = ?", (device_id,)
        ) as cursor:
            device_row = await cursor.fetchone()
            if not device_row:
                raise HTTPException(status_code=404, detail="Device not found")
            
            device = dict(device_row)
        
        # Get recent heartbeats
        async with db.execute(
            """SELECT battery_level, wifi_connected, training_active, steps_done, timestamp
               FROM heartbeats 
               WHERE device_id = ?
               ORDER BY timestamp DESC
               LIMIT 50""",
            (device_id,)
        ) as cursor:
            heartbeat_rows = await cursor.fetchall()
            
            heartbeats = []
            for row in heartbeat_rows:
                heartbeat = {
                    "battery_level": row[0],
                    "wifi_connected": bool(row[1]),
                    "training_active": bool(row[2]),
                    "steps_done": row[3],
                    "timestamp": row[4]
                }
                heartbeats.append(heartbeat)
        
        return {
            "device": {
                "device_id": device["device_id"],
                "hardware_info": device["hardware_info"],
                "os_version": device["os_version"],
                "registered_at": device["registered_at"],
                "last_seen": device["last_seen"],
                "registration_date": device["registration_date"],
                "training_enabled": bool(device["training_enabled"]),
                "learning_rate": device["learning_rate"],
                "batch_size": device["batch_size"],
                "local_epochs": device["local_epochs"]
            },
            "recent_heartbeats": heartbeats
        }

# Mount the registration server API under /api
app.mount("/api", registration_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)