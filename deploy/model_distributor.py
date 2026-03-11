#!/usr/bin/env python3
"""
Model Distributor for OysterTrain Federated Learning System
Manages global model versions and distribution via OTA updates.
"""

import os
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

@dataclass
class ModelVersion:
    version: str
    model_path: str
    size_mb: float
    timestamp: str
    checksum: str
    metadata: dict

class ModelDistributor:
    def __init__(self, models_dir: str = "deploy/models"):
        """
        Initialize the ModelDistributor.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.models_dir / "metadata.json"
        self.versions: dict = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load model metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_metadata(self):
        """Save model metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def publish_model(self, model_path: str, version: str) -> str:
        """
        Upload new model and return download URL.
        
        Args:
            model_path: Path to the model file to publish
            version: Version string for the model
            
        Returns:
            Download URL for the published model
        """
        source_path = Path(model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model filename
        model_filename = f"global_model_{version.replace('.', '_')}.pt"
        dest_path = self.models_dir / model_filename
        
        # Copy model file
        shutil.copy2(source_path, dest_path)
        
        # Calculate metadata
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        checksum = self._calculate_checksum(dest_path)
        timestamp = datetime.now().isoformat()
        
        # Store metadata
        model_info = ModelVersion(
            version=version,
            model_path=str(dest_path),
            size_mb=round(size_mb, 2),
            timestamp=timestamp,
            checksum=checksum,
            metadata={}
        )
        
        self.versions[version] = asdict(model_info)
        self._save_metadata()
        
        # Return download URL (relative path for now)
        return f"/models/{model_filename}"
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        """
        Get current global model info.
        
        Returns:
            ModelVersion object or None if no models exist
        """
        if not self.versions:
            return None
        
        # Sort by timestamp and get latest
        latest_version = max(
            self.versions.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        return ModelVersion(**latest_version[1])
    
    def get_download_url(self, version: str) -> Optional[str]:
        """
        Get signed URL for model download.
        
        Args:
            version: Version string
            
        Returns:
            Download URL or None if version not found
        """
        if version not in self.versions:
            return None
        
        model_filename = f"global_model_{version.replace('.', '_')}.pt"
        return f"/models/{model_filename}"
    
    def cleanup_old_versions(self, keep: int = 3):
        """
        Remove old model files, keeping only the most recent versions.
        
        Args:
            keep: Number of versions to keep
        """
        if len(self.versions) <= keep:
            return
        
        # Sort versions by timestamp (oldest first)
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest versions
        to_remove = sorted_versions[:-keep]
        
        for version, info in to_remove:
            model_path = Path(info['model_path'])
            if model_path.exists():
                model_path.unlink()
            del self.versions[version]
        
        self._save_metadata()
    
    def list_versions(self) -> List[ModelVersion]:
        """
        List all available model versions.
        
        Returns:
            List of ModelVersion objects sorted by timestamp (newest first)
        """
        versions = [
            ModelVersion(**info) 
            for info in self.versions.values()
        ]
        return sorted(versions, key=lambda x: x.timestamp, reverse=True)

# Example usage
if __name__ == "__main__":
    distributor = ModelDistributor()
    print("Model Distributor initialized")
    print(f"Models directory: {distributor.models_dir.absolute()}")
    
    # Example: publish a model (would need actual model file)
    # distributor.publish_model("path/to/model.pt", "1.0.0")
    # latest = distributor.get_latest_version()
    # if latest:
    #     print(f"Latest version: {latest.version}")