#!/bin/bash
# Documentation alignment checker wrapper script

set -e

echo "🚀 Jarvis Assistant Documentation Alignment Checker"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Run the documentation alignment checker
echo "🔍 Running documentation alignment checks..."
python3 scripts/check-documentation-alignment.py

echo ""
echo "✅ Documentation alignment check complete!"
echo ""
echo "💡 To fix issues:"
echo "   • Update file references to match current structure"
echo "   • Sync interface documentation with implementation"
echo "   • Refresh performance metrics with latest benchmarks"
echo ""
echo "📅 Next steps:"
echo "   • Run quarterly documentation review process"
echo "   • Update implementation status document"
echo "   • Validate cross-references and links"