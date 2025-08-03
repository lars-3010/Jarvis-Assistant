#!/usr/bin/env python3
"""
Setup script for Jarvis Assistant.

This script helps with initial project setup, dependency installation,
and configuration validation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

def check_uv_installed() -> bool:
    """Check if UV is installed."""
    try:
        result = run_command("uv --version", check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_uv() -> bool:
    """Install UV package manager."""
    print("Installing UV package manager...")
    try:
        # Use the official UV installation script
        install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        run_command(install_cmd)
        print("UV installed successfully!")
        return True
    except Exception as e:
        print(f"Failed to install UV: {e}")
        return False

def setup_environment() -> bool:
    """Setup the project environment."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create .env file if it doesn't exist
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print(f"Created {env_file}")
        print("Please edit .env file to configure your vault path and other settings.")
    
    # Create data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Created data directory: {data_dir}")
    
    return True

def install_dependencies(include_dev: bool = True, include_future: bool = False) -> bool:
    """Install project dependencies using UV."""
    print("Installing dependencies with UV...")
    
    try:
        # Sync dependencies
        if include_dev and include_future:
            run_command("uv sync --group dev --group future")
        elif include_dev:
            run_command("uv sync --group dev")
        else:
            run_command("uv sync")
        
        print("Dependencies installed successfully!")
        return True
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        return False

def validate_configuration() -> bool:
    """Validate the project configuration."""
    print("Validating configuration...")
    
    try:
        # Test importing the main module
        result = run_command("uv run python -c 'import jarvis; print(\"Import successful\")'", check=False)
        if result.returncode != 0:
            print("Warning: Failed to import jarvis module")
            print(result.stderr)
            return False
        
        # Test configuration loading
        result = run_command("uv run python -c 'from jarvis.utils.config import get_settings; print(\"Config loaded\")'", check=False)
        if result.returncode != 0:
            print("Warning: Failed to load configuration")
            print(result.stderr)
            return False
        
        print("Configuration validation successful!")
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def check_docker() -> bool:
    """Check if Docker is available for Neo4j."""
    try:
        result = run_command("docker --version", check=False)
        if result.returncode == 0:
            print("Docker is available for Neo4j services")
            return True
        else:
            print("Docker not found - you'll need to install Neo4j manually")
            return False
    except Exception:
        print("Docker not found - you'll need to install Neo4j manually")
        return False

def start_services() -> bool:
    """Start required services."""
    print("Starting services...")
    
    if check_docker():
        # Check if docker-compose services file exists
        compose_file = Path("docker/docker-compose.services.yml")
        if compose_file.exists():
            try:
                run_command(f"docker compose -f {compose_file} up -d")
                print("Neo4j service started successfully!")
                return True
            except Exception as e:
                print(f"Failed to start Neo4j service: {e}")
                return False
        else:
            print("Docker compose services file not found")
            return False
    
    return False

def main():
    """Main setup function."""
    print("=== Jarvis Assistant Setup ===")
    
    # Check and install UV
    if not check_uv_installed():
        print("UV not found. Installing...")
        if not install_uv():
            print("Failed to install UV. Please install manually:")
            print("curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)
    else:
        print("UV is already installed ✓")
    
    # Setup environment
    if not setup_environment():
        print("Failed to setup environment")
        sys.exit(1)
    
    # Install dependencies
    include_dev = "--no-dev" not in sys.argv
    include_future = "--future" in sys.argv
    
    if not install_dependencies(include_dev, include_future):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Validate configuration
    if not validate_configuration():
        print("Configuration validation failed - please check your setup")
    
    # Check Docker and optionally start services
    if "--start-services" in sys.argv:
        start_services()
    else:
        check_docker()
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Edit .env file to configure your vault path")
    print("2. Start Neo4j service: docker compose -f docker/docker-compose.services.yml up -d")
    print("3. Test the installation: uv run jarvis --help")
    print("4. Index your vault: uv run jarvis index --vault /path/to/vault")
    print("5. Start MCP server: uv run jarvis mcp --vault /path/to/vault --watch")

if __name__ == "__main__":
    main()