#!/usr/bin/env python3
"""Debug script to test MCP server startup."""

import sys
import os
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mcp_server():
    """Test the MCP server startup process."""
    print("Testing MCP server startup...", file=sys.stderr)
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_dir)
    
    # Run the MCP server command
    vault_path = os.getenv('JARVIS_VAULT_PATH')
    if not vault_path:
        print("Error: JARVIS_VAULT_PATH environment variable not set", file=sys.stderr)
        return
        
    cmd = [
        "uv", "run", "jarvis", "mcp", 
        "--vault", vault_path
    ]
    
    print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr)
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send an MCP initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "debug-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print(f"Sending init request: {json.dumps(init_request)}", file=sys.stderr)
        
        # Send the request
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Wait for response with timeout
        import select
        import time
        
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print(f"Process exited with code: {process.returncode}", file=sys.stderr)
                break
                
            # Check if there's output available
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                line = process.stdout.readline()
                if line:
                    print(f"Server response: {line.strip()}", file=sys.stderr)
                    break
        
        # Get any stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"Server stderr: {stderr_output}", file=sys.stderr)
            
        # Clean up
        process.terminate()
        process.wait()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    test_mcp_server()