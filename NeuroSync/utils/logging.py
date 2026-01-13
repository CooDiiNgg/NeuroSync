import logging
import sys
from typing import Optional

class NeuroSyncLogger:
    def __init__(self, name: str = "NeuroSync", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)
    
    def debug(self, msg: str) -> None:
        self.logger.debug(msg)
    
    def warning(self, msg: str) -> None:
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        self.logger.error(msg)
    
    def training_progress(
        self,
        episode: int,
        total: int,
        accuracy: float,
        loss: float,
        **kwargs,
    ) -> None:
        msg = f"Episode {episode}/{total} | Acc: {accuracy:.1f}% | Loss: {loss:.6f}"
        for k, v in kwargs.items():
            if isinstance(v, float):
                msg += f" | {k}: {v:.4f}"
            else:
                msg += f" | {k}: {v}"
        self.info(msg)


_logger: Optional[NeuroSyncLogger] = None

def get_logger() -> NeuroSyncLogger:
    global _logger
    if _logger is None:
        _logger = NeuroSyncLogger()
    return _logger