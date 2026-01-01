"""Main entry point for LivingWorld application."""

import logging
from src.core.logging_config import setup_logging
from src.cli.interface import main

if __name__ == "__main__":
    # Set up logging
    setup_logging(
        level=logging.INFO,
        detailed=True,
    )
    main()
