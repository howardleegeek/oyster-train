#!/usr/bin/env python3
"""
Training Scheduler for OysterTrain Federated Learning System
Manages when devices should participate in training based on constraints.
"""

import hashlib
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os

@dataclass
class TimeWindow:
    start_time: time
    end_time: time
    
    def contains(self, check_time: time) -> bool:
        """Check if a time falls within this window."""
        if self.start_time <= self.end_time:
            return self.start_time <= check_time <= self.end_time
        else:  # Window crosses midnight
            return check_time >= self.start_time or check_time <= self.end_time
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ScheduleConfig:
    quiet_hours_start: time = time(23, 0)  # 11 PM
    quiet_hours_end: time = time(7, 0)     # 7 AM
    min_battery_percent: int = 30
    max_training_hours_per_day: int = 4
    stagger_window_hours: int = 6  # Stagger training over 6 hours
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleConfig':
        # Convert time strings back to time objects
        if 'quiet_hours_start' in data and isinstance(data['quiet_hours_start'], str):
            data['quiet_hours_start'] = time.fromisoformat(data['quiet_hours_start'])
        if 'quiet_hours_end' in data and isinstance(data['quiet_hours_end'], str):
            data['quiet_hours_end'] = time.fromisoformat(data['quiet_hours_end'])
        return cls(**data)

class TrainingScheduler:
    def __init__(self, config_path: str = "deploy/schedule_config.json"):
        """
        Initialize the TrainingScheduler.
        
        Args:
            config_path: Path to schedule configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.device_windows: Dict[str, TimeWindow] = {}  # Cache device time windows
    
    def _load_config(self) -> ScheduleConfig:
        """Load schedule configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return ScheduleConfig.from_dict(data)
            except (json.JSONDecodeError, IOError):
                pass
        # Return default config
        return ScheduleConfig()
    
    def _save_config(self):
        """Save schedule configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def should_train(self, device_status: Dict[str, Any]) -> bool:
        """
        Determine if a device should train based on current status.
        
        Args:
            device_status: Dictionary containing device status information
                          Expected keys: is_charging, is_wifi, battery_level
                          
        Returns:
            True if device should train, False otherwise
        """
        # Check basic requirements
        if not device_status.get('is_charging', False):
            return False
        
        if not device_status.get('is_wifi', False):
            return False
        
        battery_level = device_status.get('battery_level', 0)
        if battery_level < self.config.min_battery_percent:
            return False
        
        # Check quiet hours
        current_time = datetime.now().time()
        if self._is_quiet_hours(current_time):
            return False
        
        # Check daily training quota (would need historical data)
        # For now, we'll assume quota is not exceeded
        # In a full implementation, we'd check training history for today
        
        return True
    
    def _is_quiet_hours(self, check_time: time) -> bool:
        """Check if current time is within quiet hours."""
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        
        if start <= end:
            # Quiet hours don't cross midnight (e.g., 22:00 to 06:00)
            return start <= check_time <= end
        else:
            # Quiet hours cross midnight (e.g., 23:00 to 07:00)
            return check_time >= start or check_time <= end
    
    def get_training_window(self, device_id: str) -> TimeWindow:
        """
        Get assigned training window for a device to stagger training times.
        
        Args:
            device_id: Unique identifier for the device
            
        Returns:
            TimeWindow object representing the device's training window
        """
        # Return cached window if exists
        if device_id in self.device_windows:
            return self.device_windows[device_id]
        
        # Generate deterministic window based on device_id hash
        hash_int = int(hashlib.md5(device_id.encode()).hexdigest(), 16)
        # Map hash to offset within stagger window (0 to stagger_window_hours-1)
        offset_hours = hash_int % self.config.stagger_window_hours
        
        # Define base training window (e.g., 8 PM to 2 AM next day)
        base_start = time(20, 0)  # 8 PM
        base_end = time(2, 0)     # 2 AM (next day)
        
        # Apply offset to stagger windows
        start_time = self._add_hours_to_time(base_start, offset_hours)
        end_time = self._add_hours_to_time(base_end, offset_hours)
        
        window = TimeWindow(start_time=start_time, end_time=end_time)
        self.device_windows[device_id] = window
        
        return window
    
    def _add_hours_to_time(self, time_obj: time, hours: int) -> time:
        """Add hours to a time object, handling day overflow."""
        # Convert to datetime, add hours, convert back to time
        dummy_date = datetime(2023, 1, 1)
        dt = datetime.combine(dummy_date, time_obj)
        dt = dt + timedelta(hours=hours)
        return dt.time()
    
    def update_schedule(self, new_config: Dict[str, Any]):
        """
        Update global schedule configuration.
        
        Args:
            new_config: Dictionary containing new configuration values
        """
        # Update config with new values
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Save updated config
        self._save_config()
        
        # Clear device window cache as it depends on config
        self.device_windows.clear()
    
    def get_device_training_status(self, device_id: str, device_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive training status for a device.
        
        Args:
            device_id: Device identifier
            device_status: Current device status
            
        Returns:
            Dictionary with training status information
        """
        should_train = self.should_train(device_status)
        training_window = self.get_training_window(device_id)
        current_time = datetime.now().time()
        in_window = training_window.contains(current_time)
        
        return {
            'device_id': device_id,
            'should_train_based_on_status': should_train,
            'in_training_window': in_window,
            'can_train_now': should_train and in_window,
            'training_window': training_window.to_dict(),
            'current_time': current_time.isoformat(),
            'quiet_hours': {
                'start': self.config.quiet_hours_start.isoformat(),
                'end': self.config.quiet_hours_end.isoformat(),
                'currently_in_quiet_hours': self._is_quiet_hours(current_time)
            }
        }

# Example usage
if __name__ == "__main__":
    scheduler = TrainingScheduler()
    print("Training Scheduler initialized")
    
    # Example device status
    device_status = {
        'is_charging': True,
        'is_wifi': True,
        'battery_level': 75
    }
    
    # Check if device should train
    should_train = scheduler.should_train(device_status)
    print(f"Device should train: {should_train}")
    
    # Get training window for a device
    device_id = "test_device_123"
    window = scheduler.get_training_window(device_id)
    print(f"Device {device_id} training window: {window.start_time} - {window.end_time}")