#!/usr/bin/env python3
"""
Dataset Generation Tool Deployment Script

This script handles the deployment and setup of the dataset generation tool
within the Jarvis Assistant system. It ensures proper integration without
affecting core functionality.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Try to import Jarvis components, but handle gracefully if not available
try:
    from jarvis.utils.config import get_settings, JarvisSettings
    from jarvis.utils.logging import setup_logging
    logger = setup_logging(__name__)
    JARVIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Jarvis components not fully available: {e}")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    JARVIS_AVAILABLE = False


class DeploymentError(Exception):
    """Raised when deployment fails."""
    pass


class DatasetToolDeployment:
    """Handles deployment of the dataset generation tool."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_root = self.project_root / "src"
        self.tool_path = self.src_root / "jarvis" / "tools" / "dataset_generation"
        self.backup_dir = None
        
    def verify_prerequisites(self) -> List[str]:
        """Verify that all prerequisites are met for deployment."""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 11):
            issues.append(f"Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check if we're in the correct directory structure
        if not self.src_root.exists():
            issues.append(f"Source directory not found: {self.src_root}")
        
        if not (self.src_root / "jarvis").exists():
            issues.append("Jarvis package not found in src directory")
        
        # Check if core services exist
        required_services = [
            "jarvis/services/vault/reader.py",
            "jarvis/services/vector/encoder.py", 
            "jarvis/services/graph/database.py"
        ]
        
        for service in required_services:
            service_path = self.src_root / service
            if not service_path.exists():
                issues.append(f"Required service not found: {service}")
        
        # Check if main.py exists and is writable
        main_py = self.src_root / "jarvis" / "main.py"
        if not main_py.exists():
            issues.append("Main CLI file not found: jarvis/main.py")
        elif not os.access(main_py, os.W_OK):
            issues.append("Main CLI file is not writable")
        
        return issues
    
    def check_tool_installation(self) -> Dict[str, Any]:
        """Check current installation status of the dataset tool."""
        status = {
            "tool_directory_exists": self.tool_path.exists(),
            "cli_command_integrated": False,
            "dependencies_available": True,
            "components": {}
        }
        
        # Check if CLI command is integrated
        main_py = self.src_root / "jarvis" / "main.py"
        if main_py.exists():
            content = main_py.read_text()
            status["cli_command_integrated"] = "generate-dataset" in content
        
        # Check individual components
        if self.tool_path.exists():
            components = [
                "dataset_generator.py",
                "models/__init__.py",
                "extractors/__init__.py", 
                "generators/__init__.py",
                "utils/__init__.py"
            ]
            
            for component in components:
                component_path = self.tool_path / component
                status["components"][component] = component_path.exists()
        
        # Check dependencies
        try:
            import pandas
            import networkx
            import numpy
            import scipy
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            status["dependencies_available"] = False
            status["missing_dependency"] = str(e)
        
        return status
    
    def create_backup(self) -> Path:
        """Create a backup of current state before deployment."""
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.project_root / f"backup_dataset_tool_{timestamp}"
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating backup at: {self.backup_dir}")
        
        # Backup main.py
        main_py = self.src_root / "jarvis" / "main.py"
        if main_py.exists():
            shutil.copy2(main_py, self.backup_dir / "main.py")
        
        # Backup tool directory if it exists
        if self.tool_path.exists():
            shutil.copytree(self.tool_path, self.backup_dir / "dataset_generation")
        
        # Backup config if it has dataset settings
        try:
            settings = get_settings()
            config_backup = {
                "dataset_output_dir": settings.dataset_output_dir,
                "embedding_model_name": settings.embedding_model_name,
                "embedding_batch_size": settings.embedding_batch_size
            }
            with open(self.backup_dir / "config_backup.json", "w") as f:
                json.dump(config_backup, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not backup config: {e}")
        
        return self.backup_dir
    
    def verify_installation(self) -> Dict[str, Any]:
        """Verify that the tool is properly installed and functional."""
        verification_results = {
            "success": True,
            "tests_passed": [],
            "tests_failed": [],
            "warnings": []
        }
        
        # Test 1: Import test
        try:
            from jarvis.tools.dataset_generation import DatasetGenerator
            verification_results["tests_passed"].append("Import test: DatasetGenerator")
        except ImportError as e:
            verification_results["tests_failed"].append(f"Import test failed: {e}")
            verification_results["success"] = False
        
        # Test 2: CLI command test
        try:
            result = subprocess.run([
                sys.executable, "-m", "jarvis.main", "generate-dataset", "--help"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                verification_results["tests_passed"].append("CLI command test: generate-dataset --help")
            else:
                verification_results["tests_failed"].append(f"CLI command test failed: {result.stderr}")
                verification_results["success"] = False
        except Exception as e:
            verification_results["tests_failed"].append(f"CLI command test error: {e}")
            verification_results["success"] = False
        
        # Test 3: Service integration test
        try:
            from jarvis.services.vault.reader import VaultReader
            from jarvis.services.vector.encoder import VectorEncoder
            from jarvis.services.graph.database import GraphDatabase
            verification_results["tests_passed"].append("Service integration test: Core services importable")
        except ImportError as e:
            verification_results["tests_failed"].append(f"Service integration test failed: {e}")
            verification_results["success"] = False
        
        # Test 4: Configuration test
        if JARVIS_AVAILABLE:
            try:
                settings = get_settings()
                dataset_path = settings.get_dataset_output_path()
                if dataset_path.exists() or dataset_path.parent.exists():
                    verification_results["tests_passed"].append("Configuration test: Dataset output path accessible")
                else:
                    verification_results["warnings"].append("Dataset output directory parent does not exist")
            except Exception as e:
                verification_results["tests_failed"].append(f"Configuration test failed: {e}")
                verification_results["success"] = False
        else:
            verification_results["warnings"].append("Configuration test skipped: Jarvis components not available")
        
        # Test 5: Dependencies test
        try:
            import pandas
            import networkx
            import numpy
            import scipy
            verification_results["tests_passed"].append("Dependencies test: All required packages available")
        except ImportError as e:
            verification_results["tests_failed"].append(f"Dependencies test failed: {e}")
            verification_results["success"] = False
        
        return verification_results
    
    def deploy(self, force: bool = False) -> bool:
        """Deploy the dataset generation tool."""
        logger.info("Starting dataset generation tool deployment...")
        
        # Check prerequisites
        issues = self.verify_prerequisites()
        if issues:
            logger.error("Prerequisites not met:")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise DeploymentError("Prerequisites not met")
        
        # Check current installation
        current_status = self.check_tool_installation()
        if current_status["tool_directory_exists"] and not force:
            logger.info("Tool already appears to be installed. Use --force to reinstall.")
            return True
        
        # Create backup
        backup_path = self.create_backup()
        logger.info(f"Backup created at: {backup_path}")
        
        try:
            # Ensure tool directory exists
            self.tool_path.mkdir(parents=True, exist_ok=True)
            
            # The tool should already be in place from the spec implementation
            # This deployment script mainly verifies and validates the installation
            
            # Verify installation
            verification = self.verify_installation()
            if not verification["success"]:
                logger.error("Installation verification failed:")
                for failure in verification["tests_failed"]:
                    logger.error(f"  - {failure}")
                raise DeploymentError("Installation verification failed")
            
            logger.info("Dataset generation tool deployed successfully!")
            logger.info("Verification results:")
            for test in verification["tests_passed"]:
                logger.info(f"  ✓ {test}")
            
            if verification["warnings"]:
                logger.info("Warnings:")
                for warning in verification["warnings"]:
                    logger.warning(f"  ⚠ {warning}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            # Restore from backup if available
            if backup_path and backup_path.exists():
                logger.info("Attempting to restore from backup...")
                try:
                    self._restore_from_backup(backup_path)
                    logger.info("Restored from backup successfully")
                except Exception as restore_error:
                    logger.error(f"Backup restoration failed: {restore_error}")
            raise
    
    def _restore_from_backup(self, backup_path: Path):
        """Restore from backup directory."""
        # Restore main.py
        backup_main = backup_path / "main.py"
        if backup_main.exists():
            shutil.copy2(backup_main, self.src_root / "jarvis" / "main.py")
        
        # Remove tool directory and restore from backup
        if self.tool_path.exists():
            shutil.rmtree(self.tool_path)
        
        backup_tool = backup_path / "dataset_generation"
        if backup_tool.exists():
            shutil.copytree(backup_tool, self.tool_path)


def main():
    """Main deployment script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy dataset generation tool")
    parser.add_argument("--force", action="store_true", 
                       help="Force deployment even if tool appears to be installed")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify installation, don't deploy")
    parser.add_argument("--status", action="store_true",
                       help="Show current installation status")
    
    args = parser.parse_args()
    
    deployment = DatasetToolDeployment()
    
    try:
        if args.status:
            status = deployment.check_tool_installation()
            print("Dataset Generation Tool Status:")
            print(f"  Tool Directory Exists: {status['tool_directory_exists']}")
            print(f"  CLI Command Integrated: {status['cli_command_integrated']}")
            print(f"  Dependencies Available: {status['dependencies_available']}")
            
            if status["components"]:
                print("  Components:")
                for component, exists in status["components"].items():
                    print(f"    {component}: {'✓' if exists else '✗'}")
            
            return
        
        if args.verify_only:
            verification = deployment.verify_installation()
            print("Installation Verification Results:")
            print(f"  Overall Success: {verification['success']}")
            
            if verification["tests_passed"]:
                print("  Tests Passed:")
                for test in verification["tests_passed"]:
                    print(f"    ✓ {test}")
            
            if verification["tests_failed"]:
                print("  Tests Failed:")
                for test in verification["tests_failed"]:
                    print(f"    ✗ {test}")
            
            if verification["warnings"]:
                print("  Warnings:")
                for warning in verification["warnings"]:
                    print(f"    ⚠ {warning}")
            
            sys.exit(0 if verification["success"] else 1)
        
        # Deploy
        success = deployment.deploy(force=args.force)
        sys.exit(0 if success else 1)
        
    except DeploymentError as e:
        print(f"Deployment Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()