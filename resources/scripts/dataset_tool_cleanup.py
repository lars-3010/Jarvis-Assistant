#!/usr/bin/env python3
"""
Dataset Generation Tool Cleanup Script

This script safely removes the dataset generation tool from the Jarvis Assistant
system without affecting core functionality. It provides options for complete
removal or selective cleanup.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import tempfile
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Try to import Jarvis components, but handle gracefully if not available
try:
    from jarvis.utils.logging import setup_logging
    logger = setup_logging(__name__)
    JARVIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Jarvis components not fully available: {e}")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    JARVIS_AVAILABLE = False


class CleanupError(Exception):
    """Raised when cleanup fails."""
    pass


class DatasetToolCleanup:
    """Handles cleanup and removal of the dataset generation tool."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_root = self.project_root / "src"
        self.tool_path = self.src_root / "jarvis" / "tools" / "dataset_generation"
        self.backup_dir = None
        
    def analyze_installation(self) -> Dict[str, Any]:
        """Analyze what components of the dataset tool are currently installed."""
        analysis = {
            "tool_directory": {
                "exists": self.tool_path.exists(),
                "path": str(self.tool_path),
                "size_mb": 0,
                "file_count": 0
            },
            "cli_integration": {
                "command_exists": False,
                "import_exists": False,
                "lines_to_remove": []
            },
            "config_settings": {
                "has_dataset_settings": False,
                "settings_found": []
            },
            "generated_files": {
                "datasets_found": [],
                "logs_found": [],
                "cache_found": []
            },
            "dependencies": {
                "tool_specific": [],
                "shared_with_core": []
            }
        }
        
        # Analyze tool directory
        if self.tool_path.exists():
            total_size = 0
            file_count = 0
            for file_path in self.tool_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            analysis["tool_directory"]["size_mb"] = total_size / (1024 * 1024)
            analysis["tool_directory"]["file_count"] = file_count
        
        # Analyze CLI integration
        main_py = self.src_root / "jarvis" / "main.py"
        if main_py.exists():
            content = main_py.read_text()
            lines = content.split('\n')
            
            # Look for dataset-related imports and commands
            for i, line in enumerate(lines):
                if "dataset_generation" in line.lower() or "generate-dataset" in line:
                    analysis["cli_integration"]["lines_to_remove"].append((i + 1, line.strip()))
                    
                    if "import" in line and "dataset_generation" in line:
                        analysis["cli_integration"]["import_exists"] = True
                    elif "generate-dataset" in line:
                        analysis["cli_integration"]["command_exists"] = True
        
        # Analyze config settings
        try:
            from jarvis.utils.config import get_settings
            settings = get_settings()
            
            # Check for dataset-specific settings
            dataset_settings = [
                ("dataset_output_dir", getattr(settings, "dataset_output_dir", None)),
            ]
            
            for setting_name, setting_value in dataset_settings:
                if setting_value:
                    analysis["config_settings"]["has_dataset_settings"] = True
                    analysis["config_settings"]["settings_found"].append({
                        "name": setting_name,
                        "value": str(setting_value)
                    })
        except Exception as e:
            logger.warning(f"Could not analyze config settings: {e}")
        
        # Look for generated files
        try:
            from jarvis.utils.config import get_settings
            settings = get_settings()
            dataset_output_path = settings.get_dataset_output_path()
            
            if dataset_output_path.exists():
                for file_path in dataset_output_path.rglob("*.csv"):
                    if "dataset" in file_path.name.lower():
                        analysis["generated_files"]["datasets_found"].append(str(file_path))
        except Exception as e:
            logger.warning(f"Could not analyze generated files: {e}")
        
        # Look for logs
        log_patterns = ["*dataset*", "*generation*"]
        for pattern in log_patterns:
            for log_file in self.project_root.rglob(pattern):
                if log_file.is_file() and log_file.suffix in ['.log', '.txt']:
                    analysis["generated_files"]["logs_found"].append(str(log_file))
        
        return analysis
    
    def create_cleanup_backup(self) -> Path:
        """Create a backup before cleanup for potential restoration."""
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.project_root / f"cleanup_backup_dataset_tool_{timestamp}"
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating cleanup backup at: {self.backup_dir}")
        
        # Backup main.py
        main_py = self.src_root / "jarvis" / "main.py"
        if main_py.exists():
            shutil.copy2(main_py, self.backup_dir / "main.py")
        
        # Backup entire tool directory
        if self.tool_path.exists():
            shutil.copytree(self.tool_path, self.backup_dir / "dataset_generation")
        
        # Backup any generated datasets
        try:
            from jarvis.utils.config import get_settings
            settings = get_settings()
            dataset_output_path = settings.get_dataset_output_path()
            
            if dataset_output_path.exists():
                dataset_backup_dir = self.backup_dir / "generated_datasets"
                dataset_backup_dir.mkdir(exist_ok=True)
                
                for file_path in dataset_output_path.rglob("*.csv"):
                    if "dataset" in file_path.name.lower():
                        shutil.copy2(file_path, dataset_backup_dir / file_path.name)
        except Exception as e:
            logger.warning(f"Could not backup generated datasets: {e}")
        
        return self.backup_dir
    
    def remove_cli_integration(self, dry_run: bool = False) -> List[str]:
        """Remove CLI command integration from main.py."""
        changes_made = []
        main_py = self.src_root / "jarvis" / "main.py"
        
        if not main_py.exists():
            return changes_made
        
        content = main_py.read_text()
        lines = content.split('\n')
        new_lines = []
        
        # Track if we're inside the generate-dataset command function
        in_command_function = False
        command_indent_level = 0
        skip_line = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip dataset-related imports
            if ("from jarvis.tools.dataset_generation" in line or 
                "import" in line and "dataset_generation" in line):
                changes_made.append(f"Removed import: {line.strip()}")
                skip_line = True
            
            # Detect start of generate-dataset command
            elif "@cli.command('generate-dataset')" in line or "def generate_dataset(" in line:
                in_command_function = True
                command_indent_level = len(line) - len(line.lstrip())
                changes_made.append(f"Removing generate-dataset command starting at line {i+1}")
                skip_line = True
            
            # If we're in the command function, skip lines until we're out
            elif in_command_function:
                current_indent = len(line) - len(line.lstrip()) if line.strip() else float('inf')
                
                # If we hit a line with same or less indentation (and it's not empty), we're out
                if line.strip() and current_indent <= command_indent_level:
                    in_command_function = False
                    # Don't skip this line, it's the start of the next function/section
                    skip_line = False
                else:
                    skip_line = True
            
            if not skip_line:
                new_lines.append(line)
            
            skip_line = False
            i += 1
        
        # Write back the modified content
        if not dry_run and changes_made:
            new_content = '\n'.join(new_lines)
            main_py.write_text(new_content)
            logger.info(f"Updated {main_py}")
        
        return changes_made
    
    def remove_tool_directory(self, dry_run: bool = False) -> bool:
        """Remove the tool directory and all its contents."""
        if not self.tool_path.exists():
            return False
        
        if not dry_run:
            shutil.rmtree(self.tool_path)
            logger.info(f"Removed tool directory: {self.tool_path}")
        else:
            logger.info(f"Would remove tool directory: {self.tool_path}")
        
        return True
    
    def clean_generated_files(self, dry_run: bool = False, keep_datasets: bool = True) -> List[str]:
        """Clean up generated files (datasets, logs, cache)."""
        cleaned_files = []
        
        if not keep_datasets:
            try:
                from jarvis.utils.config import get_settings
                settings = get_settings()
                dataset_output_path = settings.get_dataset_output_path()
                
                if dataset_output_path.exists():
                    for file_path in dataset_output_path.rglob("*.csv"):
                        if "dataset" in file_path.name.lower():
                            if not dry_run:
                                file_path.unlink()
                            cleaned_files.append(str(file_path))
                            logger.info(f"{'Would remove' if dry_run else 'Removed'} dataset: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean generated datasets: {e}")
        
        # Clean up any dataset-related log files
        log_patterns = ["*dataset*generation*", "*dataset*tool*"]
        for pattern in log_patterns:
            for log_file in self.project_root.rglob(pattern):
                if log_file.is_file() and log_file.suffix in ['.log', '.txt']:
                    if not dry_run:
                        log_file.unlink()
                    cleaned_files.append(str(log_file))
                    logger.info(f"{'Would remove' if dry_run else 'Removed'} log file: {log_file}")
        
        return cleaned_files
    
    def verify_core_functionality(self) -> Dict[str, Any]:
        """Verify that core Jarvis functionality still works after cleanup."""
        verification = {
            "success": True,
            "tests_passed": [],
            "tests_failed": []
        }
        
        # Test 1: Core imports still work
        try:
            from jarvis.services.vault.reader import VaultReader
            from jarvis.services.vector.encoder import VectorEncoder
            from jarvis.utils.config import get_settings
            verification["tests_passed"].append("Core imports test")
        except ImportError as e:
            verification["tests_failed"].append(f"Core imports failed: {e}")
            verification["success"] = False
        
        # Test 2: CLI still works
        try:
            result = subprocess.run([
                sys.executable, "-m", "jarvis.main", "--help"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                verification["tests_passed"].append("CLI help test")
            else:
                verification["tests_failed"].append(f"CLI help failed: {result.stderr}")
                verification["success"] = False
        except Exception as e:
            verification["tests_failed"].append(f"CLI test error: {e}")
            verification["success"] = False
        
        # Test 3: MCP server can start (dry run)
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                "from jarvis.mcp.server import run_mcp_server; print('MCP server import successful')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                verification["tests_passed"].append("MCP server import test")
            else:
                verification["tests_failed"].append(f"MCP server import failed: {result.stderr}")
                verification["success"] = False
        except Exception as e:
            verification["tests_failed"].append(f"MCP server test error: {e}")
            verification["success"] = False
        
        return verification
    
    def cleanup(self, 
                remove_cli: bool = True,
                remove_directory: bool = True, 
                clean_files: bool = True,
                keep_datasets: bool = True,
                dry_run: bool = False) -> Dict[str, Any]:
        """Perform cleanup of the dataset generation tool."""
        
        logger.info(f"Starting dataset tool cleanup (dry_run={dry_run})...")
        
        # Analyze current installation
        analysis = self.analyze_installation()
        logger.info("Current installation analysis:")
        logger.info(f"  Tool directory exists: {analysis['tool_directory']['exists']}")
        logger.info(f"  CLI command exists: {analysis['cli_integration']['command_exists']}")
        logger.info(f"  Generated datasets found: {len(analysis['generated_files']['datasets_found'])}")
        
        # Create backup
        if not dry_run:
            backup_path = self.create_cleanup_backup()
            logger.info(f"Backup created at: {backup_path}")
        
        cleanup_results = {
            "success": True,
            "actions_taken": [],
            "errors": [],
            "backup_path": str(self.backup_dir) if self.backup_dir else None
        }
        
        try:
            # Remove CLI integration
            if remove_cli and analysis['cli_integration']['command_exists']:
                cli_changes = self.remove_cli_integration(dry_run=dry_run)
                if cli_changes:
                    cleanup_results["actions_taken"].extend(cli_changes)
            
            # Remove tool directory
            if remove_directory and analysis['tool_directory']['exists']:
                if self.remove_tool_directory(dry_run=dry_run):
                    action = f"{'Would remove' if dry_run else 'Removed'} tool directory ({analysis['tool_directory']['file_count']} files, {analysis['tool_directory']['size_mb']:.1f} MB)"
                    cleanup_results["actions_taken"].append(action)
            
            # Clean generated files
            if clean_files:
                cleaned_files = self.clean_generated_files(dry_run=dry_run, keep_datasets=keep_datasets)
                if cleaned_files:
                    cleanup_results["actions_taken"].append(f"{'Would clean' if dry_run else 'Cleaned'} {len(cleaned_files)} generated files")
            
            # Verify core functionality still works
            if not dry_run:
                verification = self.verify_core_functionality()
                if verification["success"]:
                    cleanup_results["actions_taken"].append("Core functionality verification passed")
                else:
                    cleanup_results["errors"].extend(verification["tests_failed"])
                    cleanup_results["success"] = False
            
        except Exception as e:
            cleanup_results["errors"].append(str(e))
            cleanup_results["success"] = False
            logger.error(f"Cleanup failed: {e}")
        
        return cleanup_results


def main():
    """Main cleanup script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up dataset generation tool")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    parser.add_argument("--keep-cli", action="store_true",
                       help="Keep CLI command integration")
    parser.add_argument("--keep-directory", action="store_true", 
                       help="Keep tool directory")
    parser.add_argument("--keep-files", action="store_true",
                       help="Keep generated files")
    parser.add_argument("--remove-datasets", action="store_true",
                       help="Remove generated datasets (default: keep)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze current installation")
    
    args = parser.parse_args()
    
    cleanup = DatasetToolCleanup()
    
    try:
        if args.analyze_only:
            analysis = cleanup.analyze_installation()
            print("Dataset Generation Tool Installation Analysis:")
            print(f"\nTool Directory:")
            print(f"  Exists: {analysis['tool_directory']['exists']}")
            if analysis['tool_directory']['exists']:
                print(f"  Path: {analysis['tool_directory']['path']}")
                print(f"  Size: {analysis['tool_directory']['size_mb']:.1f} MB")
                print(f"  Files: {analysis['tool_directory']['file_count']}")
            
            print(f"\nCLI Integration:")
            print(f"  Command exists: {analysis['cli_integration']['command_exists']}")
            print(f"  Import exists: {analysis['cli_integration']['import_exists']}")
            if analysis['cli_integration']['lines_to_remove']:
                print(f"  Lines to remove: {len(analysis['cli_integration']['lines_to_remove'])}")
            
            print(f"\nGenerated Files:")
            print(f"  Datasets found: {len(analysis['generated_files']['datasets_found'])}")
            print(f"  Logs found: {len(analysis['generated_files']['logs_found'])}")
            
            if analysis['generated_files']['datasets_found']:
                print("  Dataset files:")
                for dataset in analysis['generated_files']['datasets_found']:
                    print(f"    - {dataset}")
            
            return
        
        # Perform cleanup
        results = cleanup.cleanup(
            remove_cli=not args.keep_cli,
            remove_directory=not args.keep_directory,
            clean_files=not args.keep_files,
            keep_datasets=not args.remove_datasets,
            dry_run=args.dry_run
        )
        
        print(f"Cleanup {'simulation' if args.dry_run else 'completed'}: {'Success' if results['success'] else 'Failed'}")
        
        if results["actions_taken"]:
            print("\nActions taken:")
            for action in results["actions_taken"]:
                print(f"  ✓ {action}")
        
        if results["errors"]:
            print("\nErrors encountered:")
            for error in results["errors"]:
                print(f"  ✗ {error}")
        
        if results["backup_path"] and not args.dry_run:
            print(f"\nBackup created at: {results['backup_path']}")
        
        sys.exit(0 if results["success"] else 1)
        
    except Exception as e:
        print(f"Cleanup Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()