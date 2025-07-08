#!/usr/bin/env python3
"""
DeerFlow Service Startup Script

This script provides a simple way to start the DeerFlow service with different configuration options.

Usage:
    python scripts/start_server.py --help
"""

import argparse
import os
import sys
import subprocess
import signal
import time
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_environment(env_file: str = None, performance_mode: str = "balanced"):
    """Setup environment variables"""
    # Base environment variables
    base_env = {
        "PYTHONPATH": str(project_root),
        "DEER_FLOW_ROOT": str(project_root),
    }
    
    # Performance mode configurations
    performance_configs = {
        "minimal": {
            "DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION": "false",
            "DEER_FLOW_ENABLE_COLLABORATION": "false",
            "DEER_FLOW_CONNECTION_POOL_SIZE": "10",
            "DEER_FLOW_BATCH_SIZE": "5",
            "DEER_FLOW_ENABLE_MONITORING": "false"
        },
        "balanced": {
            "DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION": "true",
            "DEER_FLOW_ENABLE_COLLABORATION": "false",
            "DEER_FLOW_CONNECTION_POOL_SIZE": "20",
            "DEER_FLOW_BATCH_SIZE": "10",
            "DEER_FLOW_ENABLE_MONITORING": "true"
        },
        "performance": {
            "DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION": "true",
            "DEER_FLOW_ENABLE_COLLABORATION": "true",
            "DEER_FLOW_CONNECTION_POOL_SIZE": "50",
            "DEER_FLOW_BATCH_SIZE": "20",
            "DEER_FLOW_ENABLE_MONITORING": "true",
            "DEER_FLOW_PARALLEL_EXECUTOR_MAX_WORKERS": "10",
            "DEER_FLOW_MEMORY_CACHE_SIZE": "1000"
        },
        "development": {
            "DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION": "true",
            "DEER_FLOW_ENABLE_COLLABORATION": "false",
            "DEER_FLOW_DEBUG_MODE": "true",
            "DEER_FLOW_CONNECTION_POOL_SIZE": "10",
            "DEER_FLOW_BATCH_SIZE": "5",
            "DEER_FLOW_ENABLE_MONITORING": "true"
        }
    }
    
    # Apply performance mode configuration
    if performance_mode in performance_configs:
        base_env.update(performance_configs[performance_mode])
        print(f"Applying performance mode: {performance_mode}")
    
    # Load configuration from environment file
    if env_file and os.path.exists(env_file):
        print(f"Loading environment file: {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    base_env[key.strip()] = value.strip().strip('"\'')
    
    # Set environment variables
    for key, value in base_env.items():
        os.environ[key] = value
        print(f"Setting environment variable: {key}={value}")

def check_dependencies():
    """Check dependencies"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "langgraph",
        "langchain"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
):
    """Start server"""
    print(f"Starting DeerFlow server...")
    print(f"Address: http://{host}:{port}")
    print(f"Workers: {workers}")
    print(f"Hot reload: {reload}")
    print(f"Log level: {log_level}")
    
    # Build uvicorn command
    cmd = [
        "uvicorn",
        "src.server.app:app",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level
    ]
    
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
    
    if reload:
        cmd.append("--reload")
        cmd.extend(["--reload-dir", "src"])
    
    # Start server
    try:
        print(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, cwd=project_root)
        
        # Wait for interrupt signal
        def signal_handler(signum, frame):
            print("\nStopping server...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for process to end
        process.wait()
        
    except FileNotFoundError:
        print("Error: uvicorn command not found")
        print("Please ensure uvicorn is installed: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DeerFlow service startup script")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Bind host address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (development mode)")
    parser.add_argument("--log-level", choices=["critical", "error", "warning", "info", "debug"], 
                       default="info", help="Log level")
    
    # Environment configuration
    parser.add_argument("--env-file", help="Environment variables file path")
    parser.add_argument("--performance-mode", 
                       choices=["minimal", "balanced", "performance", "development"],
                       default="balanced", help="Performance mode")
    
    # Other options
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration only, do not start server")
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        if not check_dependencies():
            sys.exit(1)
        if args.check_deps:
            print("All dependencies check passed")
            return
    
    # Setup environment
    setup_environment(args.env_file, args.performance_mode)
    
    # Display configuration information
    print("\n" + "="*50)
    print("DeerFlow Server Configuration")
    print("="*50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Hot reload: {args.reload}")
    print(f"Log level: {args.log_level}")
    print(f"Performance mode: {args.performance_mode}")
    if args.env_file:
        print(f"Environment file: {args.env_file}")
    print("="*50)
    
    # Display important environment variables
    important_vars = [
        "DEER_FLOW_ENABLE_ADVANCED_OPTIMIZATION",
        "DEER_FLOW_ENABLE_COLLABORATION",
        "DEER_FLOW_CONNECTION_POOL_SIZE",
        "DEER_FLOW_BATCH_SIZE",
        "DEER_FLOW_ENABLE_MONITORING"
    ]
    
    print("\nImportant Configuration:")
    for var in important_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    if args.dry_run:
        print("\nDry run mode - server not started")
        return
    
    print("\nStarting server...")
    print("Press Ctrl+C to stop server")
    print("="*50 + "\n")
    
    # Start server
    start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()