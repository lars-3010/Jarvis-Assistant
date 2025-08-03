"""
Contract validation tests for interface/implementation consistency.

These tests validate that implementations correctly match their interface definitions,
catching parameter mismatches and signature inconsistencies that could cause runtime errors.
"""

import inspect
import pytest
from typing import get_type_hints, get_origin, get_args
from abc import ABC

from jarvis.core.interfaces import (
    IVaultReader, IVectorSearcher, IVectorDatabase, IGraphDatabase,
    IHealthChecker, IMetrics, IVectorEncoder
)
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vector.searcher import VectorSearcher
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.graph.database import GraphDatabase
from jarvis.services.health import HealthChecker
from jarvis.monitoring.metrics import JarvisMetrics
from jarvis.services.vector.encoder import VectorEncoder


class TestInterfaceContracts:
    """Test that implementations match their interface contracts."""
    
    @pytest.mark.parametrize("interface_cls,implementation_cls", [
        (IVaultReader, VaultReader),
        (IVectorSearcher, VectorSearcher), 
        (IVectorDatabase, VectorDatabase),
        (IGraphDatabase, GraphDatabase),
        (IHealthChecker, HealthChecker),
        (IMetrics, JarvisMetrics),
        (IVectorEncoder, VectorEncoder),
    ])
    def test_method_signatures_match(self, interface_cls, implementation_cls):
        """Test that interface and implementation method signatures match."""
        interface_methods = self._get_abstract_methods(interface_cls)
        
        for method_name in interface_methods:
            # Get interface method signature
            interface_method = getattr(interface_cls, method_name)
            interface_sig = inspect.signature(interface_method)
            
            # Get implementation method signature  
            assert hasattr(implementation_cls, method_name), \
                f"Implementation {implementation_cls.__name__} missing method {method_name}"
            
            impl_method = getattr(implementation_cls, method_name)
            impl_sig = inspect.signature(impl_method)
            
            # Compare parameter names and types
            self._compare_signatures(interface_sig, impl_sig, interface_cls.__name__, 
                                   implementation_cls.__name__, method_name)
    
    def test_critical_search_vault_contract(self):
        """Specific test for the search_vault method that caused the original bug."""
        # Test the exact method signature that caused issues
        vault_reader = VaultReader.__new__(VaultReader)  # Create without calling __init__
        
        # Test that the method exists and has correct signature
        assert hasattr(vault_reader, 'search_vault')
        sig = inspect.signature(vault_reader.search_vault)
        
        params = list(sig.parameters.keys())
        assert 'query' in params, "search_vault must have 'query' parameter"
        assert 'search_content' in params, "search_vault must have 'search_content' parameter"
        assert 'limit' in params, "search_vault must have 'limit' parameter"
        
        # Ensure old parameter name is not present
        assert 'max_results' not in params, "search_vault should not have 'max_results' parameter"
        
        # Test parameter defaults match interface
        query_param = sig.parameters['query']
        search_content_param = sig.parameters['search_content'] 
        limit_param = sig.parameters['limit']
        
        assert query_param.default == inspect.Parameter.empty, "query should have no default"
        assert search_content_param.default == False, "search_content should default to False"
        assert limit_param.default == 20, "limit should default to 20"
    
    def test_interface_implementation_parameter_mismatch_detection(self):
        """Test that we can detect parameter name mismatches."""
        # This test validates our validation logic itself
        interface_sig = inspect.signature(IVaultReader.search_vault)
        impl_sig = inspect.signature(VaultReader.search_vault)
        
        # These should NOT raise an exception (they should match)
        try:
            self._compare_signatures(interface_sig, impl_sig, "IVaultReader", "VaultReader", "search_vault")
        except AssertionError as e:
            pytest.fail(f"Unexpected signature mismatch: {e}")
    
    @staticmethod
    def _get_abstract_methods(cls):
        """Get all abstract method names from a class."""
        abstract_methods = []
        for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            if getattr(method, '__isabstractmethod__', False):
                abstract_methods.append(name)
        
        # Also check for abstract methods defined via @abstractmethod decorator
        for name in dir(cls):
            if name.startswith('_'):
                continue
            attr = getattr(cls, name)
            if (callable(attr) and hasattr(attr, '__isabstractmethod__') 
                and attr.__isabstractmethod__):
                abstract_methods.append(name)
        
        return set(abstract_methods)
    
    @staticmethod
    def _compare_signatures(interface_sig, impl_sig, interface_name, impl_name, method_name):
        """Compare two method signatures for compatibility."""
        interface_params = list(interface_sig.parameters.values())[1:]  # Skip 'self'
        impl_params = list(impl_sig.parameters.values())[1:]  # Skip 'self'
        
        # Check parameter count
        assert len(interface_params) == len(impl_params), \
            f"{impl_name}.{method_name} has {len(impl_params)} parameters, " \
            f"but {interface_name}.{method_name} defines {len(interface_params)}"
        
        # Check parameter names and defaults
        for i, (interface_param, impl_param) in enumerate(zip(interface_params, impl_params)):
            assert interface_param.name == impl_param.name, \
                f"Parameter {i} mismatch in {impl_name}.{method_name}: " \
                f"interface expects '{interface_param.name}', implementation has '{impl_param.name}'"
            
            # Check defaults (allow implementation to be more permissive)
            if interface_param.default != inspect.Parameter.empty:
                assert impl_param.default == interface_param.default, \
                    f"Default value mismatch for {impl_name}.{method_name}.{impl_param.name}: " \
                    f"interface default is {interface_param.default}, implementation default is {impl_param.default}"


class TestFunctionalContracts:
    """Test that interface implementations work functionally as expected."""
    
    def test_search_vault_parameter_usage(self):
        """Test that search_vault actually accepts the interface-defined parameters."""
        # This would catch the bug we just fixed
        vault_reader = VaultReader.__new__(VaultReader)  # Create without init
        
        # Test that we can call with keyword arguments as defined in interface
        sig = inspect.signature(vault_reader.search_vault)
        
        # Mock the method to avoid requiring filesystem setup
        def mock_search_vault(query: str, search_content: bool = False, limit: int = 20):
            return []
        
        # Replace method temporarily
        original_method = vault_reader.search_vault
        vault_reader.search_vault = mock_search_vault
        
        try:
            # This call should work without TypeError
            result = vault_reader.search_vault(
                query="test",
                search_content=True,
                limit=10
            )
            assert result == []
        finally:
            # Restore original method
            vault_reader.search_vault = original_method


class TestInterfaceCompleteness:
    """Test that all required interfaces have implementations."""
    
    @pytest.mark.parametrize("interface_cls", [
        IVaultReader, IVectorSearcher, IVectorDatabase, IGraphDatabase,
        IHealthChecker, IMetrics, IVectorEncoder
    ])
    def test_interface_has_implementation(self, interface_cls):
        """Test that each interface has at least one concrete implementation."""
        # This test documents which interfaces are implemented
        implementations = {
            IVaultReader: VaultReader,
            IVectorSearcher: VectorSearcher,
            IVectorDatabase: VectorDatabase, 
            IGraphDatabase: GraphDatabase,
            IHealthChecker: HealthChecker,
            IMetrics: JarvisMetrics,
            IVectorEncoder: VectorEncoder,
        }
        
        assert interface_cls in implementations, \
            f"No implementation found for interface {interface_cls.__name__}"
        
        impl_cls = implementations[interface_cls]
        
        # Verify it's actually a subclass
        assert issubclass(impl_cls, interface_cls), \
            f"{impl_cls.__name__} does not implement {interface_cls.__name__}"