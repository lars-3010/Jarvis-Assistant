#!/usr/bin/env python3
"""
Simple Dataset Generation Tool Verification Script

This script provides basic verification of the dataset generation tool
installation without requiring full Jarvis dependencies.
"""

import os
import sys
from pathlib import Path


def check_file_structure():
    """Check if the basic file structure is in place."""
    project_root = Path(__file__).parent.parent.parent
    tool_path = project_root / "src" / "jarvis" / "tools" / "dataset_generation"
    
    checks = {
        "Tool directory exists": tool_path.exists(),
        "Main generator exists": (tool_path / "dataset_generator.py").exists(),
        "Models package exists": (tool_path / "models" / "__init__.py").exists(),
        "Extractors package exists": (tool_path / "extractors" / "__init__.py").exists(),
        "Generators package exists": (tool_path / "generators" / "__init__.py").exists(),
        "Utils package exists": (tool_path / "utils" / "__init__.py").exists(),
    }
    
    return checks


def check_cli_integration():
    """Check if CLI integration is in place."""
    project_root = Path(__file__).parent.parent.parent
    main_py = project_root / "src" / "jarvis" / "main.py"
    
    if not main_py.exists():
        return {"Main CLI file exists": False}
    
    content = main_py.read_text()
    
    checks = {
        "Main CLI file exists": True,
        "Dataset import present": "dataset_generation" in content,
        "Generate-dataset command present": "generate-dataset" in content,
        "CLI decorator present": "@cli.command('generate-dataset')" in content,
    }
    
    return checks


def check_documentation():
    """Check if documentation files are present."""
    project_root = Path(__file__).parent.parent.parent
    docs_root = project_root / "docs"
    
    checks = {
        "User guide exists": (docs_root / "usage" / "dataset-generation-guide.md").exists(),
        "Migration guide exists": (docs_root / "usage" / "dataset-generation-migration-guide.md").exists(),
        "Troubleshooting guide exists": (docs_root / "troubleshooting" / "dataset-generation-troubleshooting.md").exists(),
        "API reference exists": (docs_root / "api" / "dataset-generation-api.md").exists(),
    }
    
    return checks


def check_scripts():
    """Check if deployment and cleanup scripts are present."""
    project_root = Path(__file__).parent.parent.parent
    scripts_root = project_root / "resources" / "scripts"
    
    checks = {
        "Deployment script exists": (scripts_root / "dataset_tool_deployment.py").exists(),
        "Cleanup script exists": (scripts_root / "dataset_tool_cleanup.py").exists(),
        "Verification script exists": (scripts_root / "verify_dataset_tool.py").exists(),
    }
    
    return checks


def main():
    """Main verification function."""
    print("Dataset Generation Tool Verification")
    print("=" * 50)
    
    all_passed = True
    
    # Check file structure
    print("\n📁 File Structure:")
    structure_checks = check_file_structure()
    for check, passed in structure_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    # Check CLI integration
    print("\n🖥️  CLI Integration:")
    cli_checks = check_cli_integration()
    for check, passed in cli_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    # Check documentation
    print("\n📚 Documentation:")
    doc_checks = check_documentation()
    for check, passed in doc_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    # Check scripts
    print("\n🔧 Scripts:")
    script_checks = check_scripts()
    for check, passed in script_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All verification checks passed!")
        print("\nThe dataset generation tool appears to be properly installed.")
        print("\nNext steps:")
        print("1. Install dependencies: uv sync")
        print("2. Test CLI command: jarvis generate-dataset --help")
        print("3. Run full verification: python resources/scripts/dataset_tool_deployment.py --verify-only")
    else:
        print("❌ Some verification checks failed!")
        print("\nThe dataset generation tool may not be properly installed.")
        print("\nTroubleshooting:")
        print("1. Run deployment script: python resources/scripts/dataset_tool_deployment.py")
        print("2. Check the troubleshooting guide: docs/troubleshooting/dataset-generation-troubleshooting.md")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())