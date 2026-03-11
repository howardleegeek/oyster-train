#!/usr/bin/env python3
"""
Fleet Monitor for OysterTrain Federated Learning System
Collects and provides metrics about device fleet status and training progress.
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import aiosqlite

@dataclass
class HeartbeatRecord:
    device_id: str
    battery_level: int
    wifi_connected: bool
    training_active: bool
    steps_done: int
    timestamp: str
    
    @classmethod
    def from_row(cls, row) -> 'HeartbeatRecord':
        return cls(
            device_id=row[0],
            battery_level=row[1],
            wifi_connected=bool(row[2]),
            training_active=bool(row[3]),
            steps_done=row[4],
            timestamp=row[5]
        )

@dataclass
class FleetStats:
    total_registered: int
    active_now: int
    trained_today: int
    avg_battery_level: float
    wifi_connected_count: int
    total_steps_today: int
    total_bytes_uploaded: int
    convergence_curve: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FleetMonitor:
    def __init__(self, db_path: str = "deploy/devices.db"):
        """
        Initialize the FleetMonitor.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
    
    async def record_heartbeat(self, device_id: str, status: Dict[str, Any]):
        """
        Store device status from heartbeat.
        
        Args:
            device_id: Device identifier
            status: Dictionary containing device status information
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO heartbeats 
                (device_id, battery_level, wifi_connected, training_active, steps_done, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                device_id,
                status.get('battery_level', 0),
                status.get('wifi_connected', False),
                status.get('training_active', False),
                status.get('steps_done', 0),
                datetime.now().isoformat()
            ))
            await db.commit()
    
    async def get_active_devices(self) -> int:
        """
        Count devices currently training.
        
        Returns:
            Number of devices with training_active=True in last 5 minutes
        """
        five_min_ago = (datetime.now() - timedelta(minutes=5)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT COUNT(DISTINCT device_id) 
                   FROM heartbeats 
                   WHERE timestamp >= ? AND training_active = 1""",
                (five_min_ago,)
            ) as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0
    
    async def get_fleet_stats(self) -> FleetStats:
        """
        Get comprehensive fleet statistics.
        
        Returns:
            FleetStats object containing various metrics
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Total registered devices
            async with db.execute("SELECT COUNT(*) FROM devices") as cursor:
                result = await cursor.fetchone()
                total_registered = result[0] if result else 0
            
            # Active now (training in last 5 minutes)
            active_now = await self.get_active_devices()
            
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
            
            # Total bytes uploaded (placeholder - would need actual tracking)
            total_bytes_uploaded = 0
            
            # Convergence curve (placeholder - would need actual training metrics)
            convergence_curve = [0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22, 0.21, 0.20]
            
            return FleetStats(
                total_registered=total_registered,
                active_now=active_now,
                trained_today=trained_today,
                avg_battery_level=round(avg_battery_level, 2),
                wifi_connected_count=wifi_connected_count,
                total_steps_today=total_steps_today,
                total_bytes_uploaded=total_bytes_uploaded,
                convergence_curve=convergence_curve
            )
    
    async def get_device_history(self, device_id: str, limit: int = 100) -> List[HeartbeatRecord]:
        """
        Get heartbeat history for a specific device.
        
        Args:
            device_id: Device identifier
            limit: Maximum number of records to return
            
        Returns:
            List of HeartbeatRecord objects ordered by timestamp (newest first)
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT device_id, battery_level, wifi_connected, training_active, steps_done, timestamp
                   FROM heartbeats 
                   WHERE device_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (device_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [HeartbeatRecord.from_row(row) for row in rows]
    
    async def get_device_latest_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest status for a specific device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Dictionary with latest device status or None if not found
        """
        history = await self.get_device_history(device_id, limit=1)
        if not history:
            return None
        
        latest = history[0]
        return {
            'device_id': latest.device_id,
            'battery_level': latest.battery_level,
            'wifi_connected': latest.wifi_connected,
            'training_active': latest.training_active,
            'steps_done': latest.steps_done,
            'timestamp': latest.timestamp
        }
    
    async def get_training_progress(self) -> Dict[str, Any]:
        """
        Get overall training progress metrics.
        
        Returns:
            Dictionary with training progress information
        """
        # This would typically integrate with actual training metrics
        # For now, return placeholder data
        return {
            'current_round': 42,
            'total_rounds': 100,
            'global_accuracy': 0.85,
            'global_loss': 0.23,
            'participating_devices': await self.get_active_devices(),
            'avg_steps_per_device': 150
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        monitor = FleetMonitor()
        print("Fleet Monitor initialized")
        
        # Example: record a heartbeat
        await monitor.record_heartbeat("test_device_001", {
            'battery_level': 85,
            'wifi_connected': True,
            'training_active': True,
            'steps_done': 1250
        })
        print("Recorded heartbeat for test_device_001")
        
        # Example: get fleet stats
        stats = await monitor.get_fleet_stats()
        print(f"Fleet stats: {stats}")
    
    asyncio.run(main())