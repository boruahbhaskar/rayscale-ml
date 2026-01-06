#!/usr/bin/env python3
"""Start local model serving API."""

import argparse

import uvicorn

from src.config import settings
from src.utils.logging import configure_logging


def serve_api(
    host: str = "0.0.0.0", port: int = 8000, reload: bool = False, workers: int = 1
) -> None:
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
        workers: Number of worker processes.
    """

    uvicorn.run(
        "src.serving.api:app",
        host=host,
        port=port,
        reload=reload and settings.environment == "development",
        workers=workers if not reload else 1,
        log_level="info" if settings.environment == "development" else "warning",
        access_log=settings.environment == "development",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start local model serving API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(log_level=args.log_level)

    print("Starting RayScale ML Platform API")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Environment: {settings.environment}")
    print(f"  Reload: {args.reload}")
    print(f"  Workers: {args.workers}")
    print()
    print("Endpoints:")
    print(f"  API: http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print()

    try:
        serve_api(
            host=args.host, port=args.port, reload=args.reload, workers=args.workers
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server failed: {e}")
        raise


if __name__ == "__main__":
    main()
