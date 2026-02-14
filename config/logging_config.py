"""
Logging configuration for Properchy.

This module sets up structured logging with proper formatting,
handlers, and log levels for different components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_to_console: Whether to log to console
    """
    config = get_settings()
    
    # Use provided values or fall back to config
    level = log_level or config.logging.LOG_LEVEL
    file_path = log_file or config.logging.LOG_FILE
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.logging.LOG_FORMAT)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_path or config.logging.LOG_TO_FILE:
        if file_path is None:
            file_path = Path("logs") / "properchy.log"
        
        # Create logs directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Setup logging on module import
setup_logging()
