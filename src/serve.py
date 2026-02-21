
import argparse

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the Delivery Delay Prediction API server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Starting server on {args.host}:{args.port} …")
    print(f"API docs available at http://localhost:{args.port}/docs")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
