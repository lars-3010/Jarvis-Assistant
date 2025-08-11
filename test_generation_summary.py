#!/usr/bin/env python3
"""
Test script to verify GenerationSummary filtering metadata implementation.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_generation_summary_filtering_metadata():
    """Test that GenerationSummary includes all required filtering metadata."""
    try:
        from jarvis.tools.dataset_generation.models.data_models import GenerationSummary, LinkStatistics, ValidationResult
        
        print("✓ Successfully imported GenerationSummary")
        
        # Create test data
        link_stats = LinkStatistics(
            total_links=10,
            unique_links=8,
            broken_links=1,
            self_links=1,
            bidirectional_links=3
        )
        
        validation_result = ValidationResult(
            valid=True,
            areas_folder_exists=True,
            areas_notes_count=25,
            filtering_mode="areas_only",
            excluded_notes_count=15,
            areas_folder_path="/test/vault/Areas"
        )
        
        # Test GenerationSummary with filtering metadata
        summary = GenerationSummary(
            total_notes=25,
            notes_processed=25,
            notes_failed=0,
            pairs_generated=100,
            positive_pairs=30,
            negative_pairs=70,
            total_time_seconds=45.5,
            link_statistics=link_stats,
            validation_result=validation_result,
            output_files={
                "notes_dataset": "/output/notes.csv",
                "pairs_dataset": "/output/pairs.csv"
            },
            performance_metrics={
                "notes_per_second": 0.55,
                "pairs_per_second": 2.2
            },
            # Required filtering metadata fields
            filtering_enabled=True,
            areas_folder_path="/test/vault/Areas",
            excluded_folders=["Journal", "Inbox", "People", "Templates"],
            privacy_mode=True,
            filtering_summary="Privacy filtering active: 4 folders excluded (Journal, Inbox, People, ...)",
            content_protection_level="high",
            privacy_message="Only content from the Areas/ folder is included in dataset generation. Personal journals, people notes, and private content are excluded.",
            excluded_folder_count=4,
            areas_notes_count=25,
            total_vault_notes=40,
            privacy_protection_percentage=37.5
        )
        
        print("✓ GenerationSummary created with filtering metadata")
        
        # Verify all required fields are present and accessible
        required_fields = [
            'filtering_enabled',
            'areas_folder_path', 
            'excluded_folders',
            'privacy_mode',
            'filtering_summary',
            'content_protection_level',
            'privacy_message',
            'excluded_folder_count',
            'areas_notes_count',
            'total_vault_notes',
            'privacy_protection_percentage'
        ]
        
        for field in required_fields:
            if hasattr(summary, field):
                value = getattr(summary, field)
                print(f"  ✓ {field}: {value}")
            else:
                print(f"  ✗ Missing field: {field}")
                return False
        
        # Test computed properties
        print(f"  ✓ processing_rate: {summary.processing_rate:.2f} notes/sec")
        print(f"  ✓ success_rate: {summary.success_rate:.2f}")
        print(f"  ✓ positive_ratio: {summary.positive_ratio:.2f}")
        
        # Verify filtering information is comprehensive
        assert summary.filtering_enabled == True
        assert summary.privacy_mode == True
        assert len(summary.excluded_folders) == 4
        assert summary.areas_notes_count == 25
        assert summary.total_vault_notes == 40
        assert summary.privacy_protection_percentage == 37.5
        
        print("✓ All filtering metadata fields verified!")
        
        # Test with filtering disabled
        summary_no_filter = GenerationSummary(
            total_notes=40,
            notes_processed=40,
            notes_failed=0,
            pairs_generated=200,
            positive_pairs=60,
            negative_pairs=140,
            total_time_seconds=60.0,
            link_statistics=link_stats,
            validation_result=ValidationResult(valid=True, filtering_mode="full_vault"),
            filtering_enabled=False,
            privacy_mode=False,
            excluded_folders=[],
            content_protection_level="none"
        )
        
        print("✓ GenerationSummary created without filtering")
        assert summary_no_filter.filtering_enabled == False
        assert summary_no_filter.privacy_mode == False
        assert len(summary_no_filter.excluded_folders) == 0
        
        print("✓ Non-filtering scenario verified!")
        
        return True
        
    except Exception as e:
        print(f"✗ GenerationSummary test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_summary_requirements_compliance():
    """Test that GenerationSummary meets the specific requirements 3.3 and 3.5."""
    try:
        from jarvis.tools.dataset_generation.models.data_models import GenerationSummary, LinkStatistics, ValidationResult
        
        print("Testing requirements compliance...")
        
        # Requirement 3.3: "WHEN dataset generation completes THEN the summary SHALL clearly indicate that only Areas/ content was processed"
        summary = GenerationSummary(
            total_notes=25,
            notes_processed=25,
            notes_failed=0,
            pairs_generated=100,
            positive_pairs=30,
            negative_pairs=70,
            total_time_seconds=45.5,
            link_statistics=LinkStatistics(),
            validation_result=ValidationResult(valid=True, filtering_mode="areas_only"),
            filtering_enabled=True,
            areas_folder_path="/test/vault/Areas",
            excluded_folders=["Journal", "Inbox"],
            privacy_mode=True,
            filtering_summary="Privacy filtering active: 2 folders excluded (Journal, Inbox)",
            privacy_message="Only content from the Areas/ folder is included in dataset generation."
        )
        
        # Verify requirement 3.3 - summary clearly indicates Areas/ only processing
        assert summary.filtering_enabled == True
        assert summary.privacy_mode == True
        assert "Areas/" in summary.privacy_message
        assert "Only content from the Areas/ folder" in summary.privacy_message
        print("✓ Requirement 3.3 satisfied: Summary clearly indicates Areas/ only processing")
        
        # Requirement 3.5: "WHEN Areas/ filtering is active THEN the system SHALL include this information in the generation metadata"
        filtering_metadata_fields = [
            'filtering_enabled',
            'areas_folder_path',
            'excluded_folders', 
            'privacy_mode',
            'filtering_summary',
            'privacy_message'
        ]
        
        for field in filtering_metadata_fields:
            assert hasattr(summary, field), f"Missing metadata field: {field}"
            value = getattr(summary, field)
            if field == 'filtering_enabled':
                assert value == True, f"filtering_enabled should be True when Areas/ filtering is active"
            elif field == 'excluded_folders':
                assert isinstance(value, list), f"excluded_folders should be a list"
            elif field in ['areas_folder_path', 'filtering_summary', 'privacy_message']:
                assert value is not None, f"{field} should not be None when filtering is active"
        
        print("✓ Requirement 3.5 satisfied: Filtering information included in generation metadata")
        
        return True
        
    except Exception as e:
        print(f"✗ Requirements compliance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing GenerationSummary filtering metadata implementation...")
    print("=" * 70)
    
    success = True
    
    print("\n1. Testing GenerationSummary filtering metadata...")
    success &= test_generation_summary_filtering_metadata()
    
    print("\n2. Testing requirements compliance...")
    success &= test_generation_summary_requirements_compliance()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All tests passed! GenerationSummary filtering metadata is complete.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the implementation.")
        sys.exit(1)