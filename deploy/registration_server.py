#!/usr/bin/env python3
"""
Registration Server for OysterTrain Federated Learning System
Handles device registration, configuration, model distribution, and heartbeats.
"""

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="OysterTrain Registration Server", version="1.0.0")

# Database setup
DATABASE_PATH = "deploy/devices.db"

# Pydantic models
class DeviceRegistration(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=100)
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    os_version: str = Field(..., min_length=1, max_length=50)

class HeartbeatData(BaseModel):
    device_id: str
    battery_level: int = Field(..., ge=0, le=100)
    wifi_connected: bool
    training_active: bool
    steps_done: int = Field(..., ge=0)

class DeviceConfig(BaseModel):
    device_id: str
    training_enabled: bool
    learning_rate: float
    batch_size: int
    local_epochs: int

class ModelInfo(BaseModel):
    version: str
    download_url: str
    size_mb: float
    timestamp: str

class FleetStats(BaseModel):
    total_registered: int
    active_now: int
    trained_today: int
    avg_battery_level: float
    wifi_connected_count: int
    total_steps_today: int

# Database initialization
async def init_db():
    """Initialize the SQLite database with required tables."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Devices table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                hardware_info TEXT,
                os_version TEXT,
                registered_at TIMESTAMP,
                last_seen TIMESTAMP,
                registration_date DATE,
                training_enabled BOOLEAN DEFAULT 1,
                learning_rate REAL DEFAULT 0.01,
                batch_size INTEGER DEFAULT 32,
                local_epochs INTEGER DEFAULT 5
            )
        """)
        
        # Heartbeats table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                battery_level INTEGER,
                wifi_connected BOOLEAN,
                training_active BOOLEAN,
                steps_done INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES devices (device_id)
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_devices_registration_date ON devices(registration_date)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_heartbeats_device_timestamp ON heartbeats(device_id, timestamp)")
        await db.commit()

# Dependency to get database connection
async def get_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db

# Helper functions
def is_recent_registration(device_id: str, registration_date: str) -> bool:
    """Check if device has registered today."""
    try:
        reg_date = datetime.strptime(registration_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        return reg_date == today
    except ValueError:
        return False

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    await init_db()

@app.post("/api/register")
async def register_device(
    registration: DeviceRegistration,
    db: aiosqlite.Connection = Depends(get_db)
):
    """Register a new device or update existing registration."""
    # Check rate limiting: max 1 registration per device per day
    async with db.execute(
        "SELECT registration_date FROM devices WHERE device_id = ?",
        (registration.device_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if row and is_recent_registration(registration.device_id, row[0]):
            raise HTTPException(
                status_code=429,
                detail="Device can only register once per day"
            )
    
    # Insert or update device
    now = datetime.now().isoformat()
    today = datetime.now().date().isoformat()
    
    await db.execute("""
        INSERT OR REPLACE INTO devices 
        (device_id, hardware_info, os_version, registered_at, last_seen, registration_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        registration.device_id,
        json.dumps(registration.hardware_info),
        registration.os_version,
        now,
        now,
        today
    ))
    await db.commit()
    
    return {"status": "registered", "device_id": registration.device_id}

@app.get("/api/config/{device_id}")
async def get_device_config(
    device_id: str,
    db: aiosqlite.Connection = Depends(get_db)
):
    """Get training configuration for a specific device."""
    async with db.execute(
        """SELECT device_id, training_enabled, learning_rate, batch_size, local_epochs 
           FROM devices WHERE device_id = ?""",
        (device_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Device not found")
        
        return DeviceConfig(
            device_id=row[0],
            training_enabled=bool(row[1]),
            learning_rate=row[2],
            batch_size=row[3],
            local_epochs=row[4]
        )

@app.get("/api/model/latest")
async def get_latest_model():
    """Get URL to download latest global model."""
    # This would typically integrate with model_distributor.py
    # For now, return a placeholder
    return {
        "version": "1.0.0",
        "download_url": "/models/global_model_v1.0.0.pt",
        "size_mb": 45.2,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/heartbeat")
async def receive_heartbeat(
    heartbeat: HeartbeatData,
    db: aiosqlite.Connection = Depends(get_db)
):
    """Receive status update from a device."""
    # Update device last_seen
    await db.execute(
        "UPDATE devices SET last_seen = ? WHERE device_id = ?",
        (datetime.now().isoformat(), heartbeat.device_id)
    )
    
    # Record heartbeat
    await db.execute("""
        INSERT INTO heartbeats 
        (device_id, battery_level, wifi_connected, training_active, steps_done, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        heartbeat.device_id,
        heartbeat.battery_level,
        heartbeat.wifi_connected,
        heartbeat.training_active,
        heartbeat.steps_done,
        datetime.now().isoformat()
    ))
    await db.commit()
    
    return {"status": "received"}

@app.get("/api/stats")
async def get_fleet_stats(
    db: aiosqlite.Connection = Depends(get_db)
):
    """Get fleet-wide statistics."""
    # Total registered devices
    async with db.execute("SELECT COUNT(*) FROM devices") as cursor:
        result = await cursor.fetchone()
        total_registered = result[0] if result else 0
    
    # Active now (training in last 5 minutes)
    five_min_ago = (datetime.now() - timedelta(minutes=5)).isoformat()
    async with db.execute(
        """SELECT COUNT(DISTINCT h.device_id) 
           FROM heartbeats h 
           WHERE h.timestamp >= ? AND h.training_active = 1""",
        (five_min_ago,)
    ) as cursor:
        result = await cursor.fetchone()
        active_now = result[0] if result else 0
    
    # Trained today
    today = datetime.now().date().isoformat()
    async with db.execute(
        """SELECT COUNT(DISTINCT device_id) 
           FROM heartbeats 
           WHERE DATE(timestamp) = ? AND training_active = 1""",
        (today,)
    ) as cursor:
        result = await cursor.fetchone()
        trained_today = result[0] if result else 0
    
    # Average battery level (last heartbeat per device)
    async with db.execute("""
        SELECT AVG(battery_level) 
        FROM (
            SELECT battery_level 
            FROM heartbeats 
            WHERE (device_id, timestamp) IN (
                SELECT device_id, MAX(timestamp) 
                FROM heartbeats 
                GROUP BY device_id
            )
        )
    """) as cursor:
        result = await cursor.fetchone()
        avg_battery_level = result[0] if result and result[0] is not None else 0.0
    
    # WiFi connected count
    async with db.execute("""
        SELECT COUNT(*) 
        FROM (
            SELECT wifi_connected 
            FROM heartbeats 
            WHERE (device_id, timestamp) IN (
                SELECT device_id, MAX(timestamp) 
                FROM heartbeats 
                GROUP BY device_id
            )
        ) 
        WHERE wifi_connected = 1
    """) as cursor:
        result = await cursor.fetchone()
        wifi_connected_count = result[0] if result else 0
    
    # Total steps today
    async with db.execute(
        """SELECT SUM(steps_done) 
           FROM heartbeats 
           WHERE DATE(timestamp) = ?""",
        (today,)
    ) as cursor:
        result = await cursor.fetchone()
        total_steps_today = result[0] if result and result[0] is not None else 0
    
    return FleetStats(
        total_registered=total_registered,
        active_now=active_now,
        trained_today=trained_today,
        avg_battery_level=round(avg_battery_level, 2),
        wifi_connected_count=wifi_connected_count,
        total_steps_today=total_steps_today
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)