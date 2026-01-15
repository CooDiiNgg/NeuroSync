"""
Visualizer for NeuroSync operations.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from NeuroSync.utils.timing import Timer
from NeuroSync.utils.logging import get_logger


@dataclass
class OperationLog:
    """Log entry for a NeuroSync operation."""
    timestamp: float
    operation: str
    details: Dict[str, Any]
    duration: float = 0.0


class Visualizer:
    """
    Visualizer for NeuroSync operations.
    
    Logs and displays what the system is doing for
    demonstration and debugging purposes.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logs: List[OperationLog] = []
        self.timer = Timer("Visualizer")
        self.logger = get_logger("Visualizer")
    
    def start_operation(self, operation: str, details: Dict[str, Any] = None) -> None:
        """Starts timing and logging an operation."""
        self.timer.start()
        if self.verbose:
            self.logger.info(f"\n[START {operation}]")
            if details:
                for k, v in details.items():
                    self.logger.info(f"  {k}: {v}")
    
    def end_operation(self, operation: str, details: Dict[str, Any] = None) -> None:
        """Ends timing and logging an operation."""
        duration = self.timer.stop()
        
        log = OperationLog(
            timestamp=self.timer.start_time or 0.0,
            operation=operation,
            details=details or {},
            duration=duration,
        )
        self.logs.append(log)
        
        if self.verbose:
            if details:
                for k, v in details.items():
                    self.logger.info(f"  {k}: {v}")
            self.logger.info(f"  Duration: {duration:.3f}s")
            
    def log_encryption(self, plaintext: str, ciphertext: str) -> None:
        """Logs an encryption operation."""
        if self.verbose:
            self.logger.info(f"\n[ENCRYPT]")
            self.logger.info(f"  Plaintext:  '{plaintext}'")
            self.logger.info(f"  Ciphertext: '{ciphertext}'")
    
    def log_decryption(self, ciphertext: str, plaintext: str) -> None:
        """Logs a decryption operation."""
        if self.verbose:
            self.logger.info(f"\n[DECRYPT]")
            self.logger.info(f"  Ciphertext: '{ciphertext}'")
            self.logger.info(f"  Plaintext:  '{plaintext}'")
    
    def log_key_rotation(self, old_key_hash: str, new_key_hash: str) -> None:
        """Logs a key rotation event."""
        if self.verbose:
            self.logger.info(f"\n[KEY ROTATION]")
            self.logger.info(f"  Old key hash: {old_key_hash[:16]}...")
            self.logger.info(f"  New key hash: {new_key_hash[:16]}...")
    
    def log_packet(self, packet, direction: str = "OUT") -> None:
        """Logs packet transmission or reception."""
        if self.verbose:
            self.logger.info(f"\n[PACKET {direction}]")
            self.logger.info(f"  Sequence: {packet.header.sequence_id}")
            self.logger.info(f"  Flags: {packet.header.flags}")
            self.logger.info(f"  Payload size: {len(packet.payload)} bytes")
    
    def summary(self) -> None:
        """Print operation summary."""
        self.logger.info("OPERATION SUMMARY")
        self.logger.info(f"Total operations: {len(self.logs)}")
        
        by_type = {}
        for log in self.logs:
            by_type.setdefault(log.operation, []).append(log)
        
        for op_type, logs in by_type.items():
            total_time = sum(l.duration for l in logs)
            self.logger.info(f"  {op_type}: {len(logs)} ops, {total_time:.3f}s total")
