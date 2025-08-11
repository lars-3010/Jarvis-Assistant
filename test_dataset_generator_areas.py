#!/usr/bin/env python3
"""
Test script to verify DatasetGenerator Areas/ filtering implementation.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_dataset_generator_initialization():
    """Test that DatasetGenerator can be initialized with Areas/ filtering parameters."""
    try:
        from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
        from jarvis.utils.config import get_settings
        
        print("✓ Successfully imported DatasetGenerator")
        
        # Test with default parameters (should use settings)
        test_vault_path = Path("/tmp/test_vault")
        test_vault_path.mkdir(exist_ok=True)
        
        # Create a simple test vault structure
        areas_path = test_vault_path / "Areas"
        areas_path.mkdir(exist_ok=True)
        
        # Create a test markdown file
        test_file = areas_path / "test.md"
        test_file.write_text("# Test Note\n\nThis is a test note.")
        
        print("✓ Created test vault structure")
        
        # Test initialization with default parameters
        try:
            generator = DatasetGenerator(
                vault_path=test_vault_path,
                skip_validation=True  # Skip validation for this test
            )
            print(f"✓ DatasetGenerator initialized with areas_only={generator.areas_only}")
            print(f"✓ Output directory: {generator.output_dir}")
            
        except Exception as e:
            print(f"✗ Failed to initialize DatasetGenerator with defaults: {e}")
            return False
        
        # Test initialization with explicit Areas filtering
        try:
            generator_areas = DatasetGenerator(
                vault_path=test_vault_path,
                areas_only=True,
                skip_validation=True
            )
            print(f"✓ DatasetGenerator initialized with explicit areas_only=True")
            
        except Exception as e:
            print(f"✗ Failed to initialize DatasetGenerator with areas_only=True: {e}")
            return False
        
        # Test initialization with Areas filtering disabled
        try:
            generator_full = DatasetGenerator(
                vault_path=test_vault_path,
                areas_only=False,
                skip_validation=True
            )
            print(f"✓ DatasetGenerator initialized with explicit areas_only=False")
            
        except Exception as e:
            print(f"✗ Failed to initialize DatasetGenerator with areas_only=False: {e}")
            return False
        
        # Test custom output directory
        try:
            custom_output = Path("/tmp/custom_datasets")
            generator_custom = DatasetGenerator(
                vault_path=test_vault_path,
                output_dir=custom_output,
                skip_validation=True
            )
            print(f"✓ DatasetGenerator initialized with custom output directory: {generator_custom.output_dir}")
            
        except Exception as e:
            print(f"✗ Failed to initialize DatasetGenerator with custom output: {e}")
            return False
        
        # Clean up
        import shutil
        shutil.rmtree(test_vault_path, ignore_errors=True)
        shutil.rmtree(custom_output, ignore_errors=True)
        
        print("✓ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_filtering_metadata():
    """Test that filtering metadata is correctly generated."""
    try:
        from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
        
        # Create test vault
        test_vault_path = Path("/tmp/test_vault_metadata")
        test_vault_path.mkdir(exist_ok=True)
        
        areas_path = test_vault_path / "Areas"
        areas_path.mkdir(exist_ok=True)
        
        # Create some test folders that should be excluded
        journal_path = test_vault_path / "Journal"
        journal_path.mkdir(exist_ok=True)
        
        inbox_path = test_vault_path / "Inbox"
        inbox_path.mkdir(exist_ok=True)
        
        # Create test files
        (areas_path / "test.md").write_text("# Test Note")
        (journal_path / "journal.md").write_text("# Journal Entry")
        (inbox_path / "inbox.md").write_text("# Inbox Note")
        
        # Test filtering metadata
        generator = DatasetGenerator(
            vault_path=test_vault_path,
            areas_only=True,
            skip_validation=True
        )
        
        metadata = generator._get_filtering_metadata()
        
        print(f"✓ Filtering metadata generated:")
        print(f"  - filtering_enabled: {metadata['filtering_enabled']}")
        print(f"  - areas_folder_path: {metadata['areas_folder_path']}")
        print(f"  - excluded_folders: {metadata['excluded_folders']}")
        print(f"  - privacy_mode: {metadata['privacy_mode']}")
        
        # Verify metadata
        assert metadata['filtering_enabled'] == True
        assert metadata['privacy_mode'] == True
        assert 'Journal' in metadata['excluded_folders']
        assert 'Inbox' in metadata['excluded_folders']
        
        print("✓ Filtering metadata test passed!")
        
        # Clean up
        import shutil
        shutil.rmtree(test_vault_path, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Filtering metadata test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing DatasetGenerator Areas/ filtering implementation...")
    print("=" * 60)
    
    success = True
    
    print("\n1. Testing DatasetGenerator initialization...")
    success &= test_dataset_generator_initialization()
    
    print("\n2. Testing filtering metadata...")
    success &= test_filtering_metadata()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! Implementation is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the implementation.")
        sys.exit(1)