"""
Access tracking and statistics management for the RAG chunking lab.
"""

import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any


class AccessTracker:
    """Tracks user access information including IP and User-Agent."""
    
    def __init__(self, stats_file: str = "stats.csv"):
        self.stats_file = stats_file
        self._ensure_stats_file()
    
    def _ensure_stats_file(self) -> None:
        """Ensure the stats CSV file exists with proper headers."""
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'timestamp', 'ip', 'user_agent'])
    
    def get_next_id(self) -> int:
        """Get the next available ID for a new record."""
        if not os.path.exists(self.stats_file):
            return 1
        
        with open(self.stats_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def log_access(self, ip: str, user_agent: str) -> None:
        """Log a new access record."""
        next_id = self.get_next_id()
        timestamp = datetime.now().isoformat()
        
        with open(self.stats_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([next_id, timestamp, ip, user_agent])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics from the access log."""
        if not os.path.exists(self.stats_file):
            return {
                'total_accesses': 0,
                'unique_ips': 0,
                'unique_user_agents': 0
            }
        
        ips = set()
        user_agents = set()
        total_count = 0
        
        with open(self.stats_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                ips.add(row['ip'])
                user_agents.add(row['user_agent'])
        
        return {
            'total_accesses': total_count,
            'unique_ips': len(ips),
            'unique_user_agents': len(user_agents)
        }


class StatsManager:
    """High-level statistics manager for the application."""
    
    def __init__(self, stats_file: str = "stats.csv"):
        self.tracker = AccessTracker(stats_file)
    
    def track_user_access(self, ip: Optional[str], user_agent: Optional[str]) -> bool:
        """
        Track user access if both IP and User-Agent are available.
        
        Returns:
            bool: True if access was logged, False otherwise
        """
        if ip and user_agent:
            self.tracker.log_access(ip, user_agent)
            return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of access statistics."""
        return self.tracker.get_stats()
