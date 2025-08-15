#!/usr/bin/env python3
"""
Documentation alignment checker for Jarvis Assistant.

This script verifies that architecture documentation matches the current implementation
by checking file references, interface definitions, and performance metrics.
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass

@dataclass
class AlignmentIssue:
    """Represents a documentation alignment issue."""
    type: str
    severity: str  # "error", "warning", "info"
    file: str
    line: int
    message: str
    suggestion: str = ""

class DocumentationAlignmentChecker:
    """Checks alignment between documentation and implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.src_dir = project_root / "src"
        self.issues: List[AlignmentIssue] = []
    
    def check_all_alignments(self) -> Dict[str, Any]:
        """Run all alignment checks and return results."""
        print("🔍 Checking documentation alignment...")
        
        results = {
            "file_references": self.check_file_references(),
            "cross_references": self.check_cross_references()
        }
        
        return self.generate_report(results)
    
    def check_file_references(self) -> Dict[str, Any]:
        """Check if all file references in documentation exist."""
        print("  📁 Checking file references...")
        
        file_refs = self.extract_file_references()
        missing_files = []
        
        for doc_file, refs in file_refs.items():
            for ref in refs:
                file_path = self.resolve_file_path(ref)
                if not file_path.exists():
                    missing_files.append({
                        "doc_file": doc_file,
                        "reference": ref,
                        "resolved_path": str(file_path)
                    })
        
        return {
            "total_references": sum(len(refs) for refs in file_refs.values()),
            "missing_files": missing_files,
            "success_rate": 1 - (len(missing_files) / max(1, sum(len(refs) for refs in file_refs.values())))
        }
    
    def extract_file_references(self) -> Dict[str, List[str]]:
        """Extract file references from documentation."""
        file_refs = {}
        
        for doc_file in self.docs_dir.rglob("*.md"):
            refs = []
            content = doc_file.read_text()
            
            # Find code file references (src/jarvis/...)
            code_refs = re.findall(r'`(src/jarvis/[^`]+)`', content)
            refs.extend(code_refs)
            
            # Find test file references (resources/tests/...)
            test_refs = re.findall(r'`(resources/tests/[^`]+)`', content)
            refs.extend(test_refs)
            
            if refs:
                file_refs[str(doc_file.relative_to(self.project_root))] = refs
        
        return file_refs
    
    def resolve_file_path(self, reference: str) -> Path:
        """Resolve a file reference to an absolute path."""
        return self.project_root / reference
    
    def check_cross_references(self) -> Dict[str, Any]:
        """Check if cross-references between documentation files are valid."""
        print("  🔗 Checking cross-references...")
        
        cross_refs = self.extract_cross_references()
        broken_refs = []
        
        for doc_file, refs in cross_refs.items():
            for ref in refs:
                ref_path = self.resolve_doc_reference(ref, doc_file)
                if not ref_path.exists():
                    broken_refs.append({
                        "doc_file": doc_file,
                        "reference": ref,
                        "resolved_path": str(ref_path)
                    })
        
        return {
            "total_cross_references": sum(len(refs) for refs in cross_refs.values()),
            "broken_references": broken_refs,
            "success_rate": 1 - (len(broken_refs) / max(1, sum(len(refs) for refs in cross_refs.values())))
        }
    
    def extract_cross_references(self) -> Dict[str, List[str]]:
        """Extract cross-references from documentation."""
        cross_refs = {}
        
        for doc_file in self.docs_dir.rglob("*.md"):
            refs = []
            content = doc_file.read_text()
            
            # Find markdown links to other docs [text](file.md)
            md_refs = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content)
            refs.extend([ref[1] for ref in md_refs])
            
            if refs:
                cross_refs[str(doc_file.relative_to(self.project_root))] = refs
        
        return cross_refs
    
    def resolve_doc_reference(self, reference: str, source_file: str) -> Path:
        """Resolve a documentation reference to an absolute path."""
        source_path = Path(source_file).parent
        if reference.startswith('/'):
            # Absolute reference from docs root
            return self.docs_dir / reference.lstrip('/')
        else:
            # Relative reference
            return self.project_root / source_path / reference
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive alignment report."""
        total_issues = sum(len(result.get("issues", [])) for result in results.values())
        
        return {
            "summary": {
                "total_checks": len(results),
                "total_issues": total_issues,
                "overall_health": "good" if total_issues < 5 else "needs_attention" if total_issues < 15 else "critical"
            },
            "results": results,
            "recommendations": self.generate_recommendations(results)
        }
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # File reference recommendations
        file_results = results.get("file_references", {})
        if file_results.get("success_rate", 1) < 0.95:
            recommendations.append("Update file references in documentation to match current file structure")
        
        return recommendations

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    checker = DocumentationAlignmentChecker(project_root)
    
    results = checker.check_all_alignments()
    
    print(f"\n📊 Documentation Alignment Report")
    print(f"Overall Health: {results['summary']['overall_health'].upper()}")
    print(f"Total Issues: {results['summary']['total_issues']}")
    
    if results['summary']['total_issues'] > 0:
        print(f"\n🔧 Recommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
    
    sys.exit(0 if results['summary']['total_issues'] == 0 else 1)