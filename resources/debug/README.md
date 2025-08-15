# Debug Scripts

This directory contains debugging and development scripts for Jarvis Assistant.

## Scripts

### debug_pairs_issue.py
Debug script for investigating pairs dataset filtering issues. This script:
- Creates a test vault with Areas/ content
- Tests the dataset generation workflow step by step
- Validates that Areas/ filtering works correctly for both notes and pairs datasets
- Helps identify issues with link extraction and negative sampling

Usage:
```bash
cd resources/debug
python debug_pairs_issue.py
```

### demo_error_handling.py
Demonstration script for enhanced database error handling. This script shows:
- Missing database error handling
- Permission error handling  
- Database corruption error handling
- Disk space error handling
- Database initializer integration

Usage:
```bash
cd resources/debug
python demo_error_handling.py
```

### pairs_dataset_generator_fix.py
Documentation of fixes needed for pairs dataset generator to respect Areas filtering. Contains:
- Code snippets showing the required fixes
- Fallback mechanisms for stratified sampling
- Integration points for filtered notes

This is a reference file, not an executable script.

## Running Debug Scripts

All scripts are designed to be run from the `resources/debug/` directory:

```bash
cd resources/debug
python <script_name>.py
```

The scripts automatically adjust their import paths to find the Jarvis source code.