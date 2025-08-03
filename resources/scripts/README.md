# Dataset Generation Tool Deployment Scripts

This directory contains scripts for deploying, managing, and cleaning up the dataset generation tool in Jarvis Assistant.

## Scripts Overview

### 1. `verify_dataset_tool.py`
**Purpose**: Quick verification of tool installation without requiring full dependencies.

**Usage**:
```bash
python3 resources/scripts/verify_dataset_tool.py
```

**Features**:
- ✅ Checks file structure
- ✅ Verifies CLI integration
- ✅ Confirms documentation presence
- ✅ Validates deployment scripts
- ✅ Works without full Jarvis dependencies

### 2. `dataset_tool_deployment.py`
**Purpose**: Comprehensive deployment and verification of the dataset generation tool.

**Usage**:
```bash
# Check installation status
python resources/scripts/dataset_tool_deployment.py --status

# Verify installation functionality
python resources/scripts/dataset_tool_deployment.py --verify-only

# Deploy/reinstall the tool
python resources/scripts/dataset_tool_deployment.py

# Force reinstallation
python resources/scripts/dataset_tool_deployment.py --force
```

**Features**:
- ✅ Prerequisites checking
- ✅ Installation status analysis
- ✅ Backup creation before changes
- ✅ Comprehensive verification tests
- ✅ Rollback capability on failure

### 3. `dataset_tool_cleanup.py`
**Purpose**: Safe removal of the dataset generation tool without affecting core Jarvis functionality.

**Usage**:
```bash
# Analyze current installation
python resources/scripts/dataset_tool_cleanup.py --analyze-only

# Preview cleanup actions
python resources/scripts/dataset_tool_cleanup.py --dry-run

# Perform cleanup
python resources/scripts/dataset_tool_cleanup.py

# Keep specific components
python resources/scripts/dataset_tool_cleanup.py --keep-cli --keep-files
```

**Features**:
- ✅ Installation analysis
- ✅ Selective cleanup options
- ✅ Backup creation before removal
- ✅ Core functionality verification
- ✅ Dry-run mode for safety

## Deployment Workflow

### Initial Installation

1. **Quick Verification** (no dependencies required):
   ```bash
   python3 resources/scripts/verify_dataset_tool.py
   ```

2. **Install Dependencies** (if needed):
   ```bash
   uv sync
   ```

3. **Full Verification**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py --verify-only
   ```

4. **Deploy if Needed**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py
   ```

### Maintenance and Updates

1. **Check Status**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py --status
   ```

2. **Reinstall if Issues**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py --force
   ```

3. **Test Functionality**:
   ```bash
   jarvis generate-dataset --help
   ```

### Removal

1. **Analyze Installation**:
   ```bash
   python resources/scripts/dataset_tool_cleanup.py --analyze-only
   ```

2. **Preview Cleanup**:
   ```bash
   python resources/scripts/dataset_tool_cleanup.py --dry-run
   ```

3. **Perform Cleanup**:
   ```bash
   python resources/scripts/dataset_tool_cleanup.py
   ```

4. **Verify Core Functionality**:
   ```bash
   jarvis --help
   ```

## Script Options Reference

### Deployment Script Options

| Option | Description |
|--------|-------------|
| `--status` | Show current installation status |
| `--verify-only` | Only verify installation, don't deploy |
| `--force` | Force deployment even if already installed |

### Cleanup Script Options

| Option | Description |
|--------|-------------|
| `--analyze-only` | Only analyze current installation |
| `--dry-run` | Show what would be done without doing it |
| `--keep-cli` | Keep CLI command integration |
| `--keep-directory` | Keep tool directory |
| `--keep-files` | Keep generated files |
| `--remove-datasets` | Remove generated datasets (default: keep) |

## Safety Features

### Backup System
- All scripts create timestamped backups before making changes
- Backups include:
  - Tool directory contents
  - CLI integration code
  - Generated datasets (optional)
  - Configuration settings

### Rollback Capability
- Deployment script can restore from backup on failure
- Manual restoration possible using backup directories
- Core functionality verification after changes

### Dry-Run Mode
- Cleanup script supports dry-run mode
- Shows exactly what would be changed
- Safe to run multiple times

## Troubleshooting

### Common Issues

#### "Command not found" errors
```bash
# Use full Python module path
python -m jarvis.main generate-dataset --help

# Or check if you're in the right directory
pwd
ls -la src/jarvis/main.py
```

#### Import errors
```bash
# Install dependencies
uv sync

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### Permission errors
```bash
# Check file permissions
ls -la resources/scripts/
chmod +x resources/scripts/*.py
```

### Recovery Procedures

#### Restore from Backup
```bash
# Find backup directory
ls -la backup_dataset_tool_*

# Restore manually if needed
cp -r backup_dataset_tool_TIMESTAMP/dataset_generation src/jarvis/tools/
cp backup_dataset_tool_TIMESTAMP/main.py src/jarvis/
```

#### Complete Reset
```bash
# Remove everything
python resources/scripts/dataset_tool_cleanup.py

# Reinstall from scratch
python resources/scripts/dataset_tool_deployment.py --force
```

## Integration with CI/CD

### Automated Deployment
```bash
#!/bin/bash
# deploy_dataset_tool.sh

set -e

echo "Verifying dataset tool installation..."
python3 resources/scripts/verify_dataset_tool.py

echo "Installing dependencies..."
uv sync

echo "Deploying dataset tool..."
python resources/scripts/dataset_tool_deployment.py --verify-only

echo "Testing CLI command..."
python -m jarvis.main generate-dataset --help

echo "Deployment successful!"
```

### Automated Testing
```bash
#!/bin/bash
# test_dataset_tool.sh

set -e

# Create test vault
mkdir -p test_vault
echo "# Test Note 1\n[[Test Note 2]]" > test_vault/note1.md
echo "# Test Note 2\nContent here" > test_vault/note2.md

# Test dataset generation
python -m jarvis.main generate-dataset \
    --vault test_vault \
    --output test_output \
    --batch-size 8

# Verify output
if [ -f "test_output/notes_dataset.csv" ] && [ -f "test_output/pairs_dataset.csv" ]; then
    echo "✅ Dataset generation test passed"
    rm -rf test_vault test_output
    exit 0
else
    echo "❌ Dataset generation test failed"
    rm -rf test_vault test_output
    exit 1
fi
```

## Best Practices

### Before Deployment
1. ✅ Run quick verification first
2. ✅ Ensure you have write permissions
3. ✅ Check available disk space
4. ✅ Backup important data

### During Deployment
1. ✅ Monitor output for errors
2. ✅ Don't interrupt the process
3. ✅ Check backup creation
4. ✅ Verify each step completes

### After Deployment
1. ✅ Run full verification
2. ✅ Test CLI command
3. ✅ Test with small dataset
4. ✅ Check documentation links

### Before Cleanup
1. ✅ Backup any important datasets
2. ✅ Run analysis first
3. ✅ Use dry-run mode
4. ✅ Verify what will be removed

### After Cleanup
1. ✅ Verify core functionality
2. ✅ Check for leftover files
3. ✅ Test other Jarvis commands
4. ✅ Clean up backup files (optional)

## Support

For issues with these scripts:

1. **Check the logs** for detailed error messages
2. **Run verification scripts** to diagnose problems
3. **Use dry-run modes** to preview changes
4. **Consult troubleshooting guides** in the docs directory
5. **Check backup directories** for recovery options

The deployment scripts are designed to be safe and reversible, with comprehensive error handling and backup systems to protect your Jarvis installation.