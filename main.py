"""Main entry point for LivingWorld application."""

import logging
from pathlib import Path
from datetime import datetime
from src.core.logging_config import setup_logging
from src.cli.interface import main

if __name__ == "__main__":
    # Create logs directory with timestamp
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"livingworld_{timestamp}.log"

    # Set up logging with file output
    setup_logging(
        level=logging.DEBUG,  # Use DEBUG level for extensive logging
        log_file=log_file,
        detailed=True,  # Include file names and line numbers
    )

    main()
