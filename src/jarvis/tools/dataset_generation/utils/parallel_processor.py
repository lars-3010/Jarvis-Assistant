"""
Parallel processing utilities for dataset generation.

This module provides parallel processing capabilities to speed up
independent feature computations using multiple CPU cores.
"""

import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Iterator, Dict
import queue

from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class ParallelProcessor:
    """Parallel processor for independent computations."""

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes (None for auto)
            use_processes: Whether to use processes instead of threads
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor = None
        
        logger.info(f"Initialized parallel processor: {max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}")

    def __enter__(self):
        """Enter context manager."""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.executor:
            self.executor.shutdown(wait=True)

    def map_parallel(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Apply function to items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            chunk_size: Size of chunks for processing (None for auto)
            
        Returns:
            List of results in same order as input
        """
        if not items:
            return []
        
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
        
        logger.debug(f"Processing {len(items)} items in parallel with chunk size {chunk_size}")
        
        try:
            if self.use_processes:
                # Use process pool for CPU-intensive tasks
                results = list(self.executor.map(func, items, chunksize=chunk_size))
            else:
                # Use thread pool for I/O-bound tasks
                futures = [self.executor.submit(func, item) for item in items]
                results = [future.result() for future in futures]
            
            logger.debug(f"Parallel processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise

    def map_parallel_with_progress(self, func: Callable, items: List[Any],
                                 progress_callback: Optional[Callable] = None,
                                 chunk_size: Optional[int] = None) -> List[Any]:
        """Apply function to items in parallel with progress tracking.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            progress_callback: Optional callback for progress updates
            chunk_size: Size of chunks for processing (None for auto)
            
        Returns:
            List of results in same order as input
        """
        if not items:
            return []
        
        logger.debug(f"Processing {len(items)} items in parallel with progress tracking")
        
        try:
            # Submit all tasks
            futures = [self.executor.submit(func, item) for item in items]
            
            # Collect results with progress tracking
            results = [None] * len(items)
            completed_count = 0
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    # Find the index of this future
                    future_index = futures.index(future)
                    results[future_index] = future.result()
                    completed_count += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(completed_count, len(items))
                    
                    # Log progress periodically
                    if completed_count % max(1, len(items) // 10) == 0:
                        logger.debug(f"Parallel progress: {completed_count}/{len(items)}")
                        
                except Exception as e:
                    logger.warning(f"Task failed in parallel processing: {e}")
                    # Keep None in results for failed tasks
            
            logger.debug(f"Parallel processing with progress completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing with progress failed: {e}")
            raise


class ThreadSafeCounter:
    """Thread-safe counter for parallel processing."""

    def __init__(self, initial_value: int = 0):
        """Initialize counter.
        
        Args:
            initial_value: Initial counter value
        """
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value.
        
        Args:
            amount: Amount to increment
            
        Returns:
            New counter value
        """
        with self._lock:
            self._value += amount
            return self._value

    def get_value(self) -> int:
        """Get current counter value.
        
        Returns:
            Current counter value
        """
        with self._lock:
            return self._value

    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class ParallelBatchProcessor:
    """Process batches of items in parallel with load balancing."""

    def __init__(self, max_workers: Optional[int] = None, batch_size: int = 32):
        """Initialize parallel batch processor.
        
        Args:
            max_workers: Maximum number of worker threads (None for auto)
            batch_size: Size of each batch
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)
        
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.completed_counter = ThreadSafeCounter()
        self.failed_counter = ThreadSafeCounter()
        
        logger.info(f"Initialized parallel batch processor: {max_workers} workers, "
                   f"batch size {batch_size}")

    def process_batches(self, items: List[Any], batch_func: Callable,
                       progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process items in parallel batches.
        
        Args:
            items: List of items to process
            batch_func: Function to process each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all results from all batches
        """
        if not items:
            return []
        
        # Create batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        # Reset counters
        self.completed_counter.reset()
        self.failed_counter.reset()
        
        all_results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batch tasks
                future_to_batch = {
                    executor.submit(self._process_batch_with_tracking, batch_func, batch, i): i
                    for i, batch in enumerate(batches)
                }
                
                # Collect results
                batch_results = [None] * len(batches)
                
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    
                    try:
                        batch_result = future.result()
                        batch_results[batch_index] = batch_result
                        
                        # Update progress
                        completed_batches = sum(1 for r in batch_results if r is not None)
                        if progress_callback:
                            # Estimate progress based on completed batches
                            estimated_items = completed_batches * self.batch_size
                            progress_callback(min(estimated_items, len(items)), len(items))
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_index} failed: {e}")
                        batch_results[batch_index] = []  # Empty result for failed batch
                
                # Flatten results
                for batch_result in batch_results:
                    if batch_result is not None:
                        all_results.extend(batch_result)
            
            logger.info(f"Parallel batch processing completed: {len(all_results)} total results, "
                       f"{self.completed_counter.get_value()} successful items, "
                       f"{self.failed_counter.get_value()} failed items")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Parallel batch processing failed: {e}")
            raise

    def _process_batch_with_tracking(self, batch_func: Callable, batch: List[Any], 
                                   batch_index: int) -> List[Any]:
        """Process a single batch with tracking.
        
        Args:
            batch_func: Function to process the batch
            batch: List of items in the batch
            batch_index: Index of the batch
            
        Returns:
            List of results from the batch
        """
        try:
            logger.debug(f"Processing batch {batch_index}: {len(batch)} items")
            
            batch_results = batch_func(batch)
            
            # Count successful and failed items
            successful = sum(1 for r in batch_results if r is not None)
            failed = len(batch_results) - successful
            
            self.completed_counter.increment(successful)
            self.failed_counter.increment(failed)
            
            logger.debug(f"Batch {batch_index} completed: {successful} successful, {failed} failed")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch {batch_index} processing failed: {e}")
            self.failed_counter.increment(len(batch))
            return [None] * len(batch)  # Return None for all items in failed batch


class ParallelFeatureExtractor:
    """Specialized parallel processor for feature extraction."""

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel feature extractor.
        
        Args:
            max_workers: Maximum number of worker threads (None for auto)
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 6)  # Conservative for feature extraction
        
        self.max_workers = max_workers
        self.processor = ParallelProcessor(max_workers, use_processes=False)  # Use threads for I/O
        
        logger.info(f"Initialized parallel feature extractor: {max_workers} workers")

    def extract_features_parallel(self, items: List[Any], 
                                 feature_func: Callable,
                                 progress_callback: Optional[Callable] = None) -> List[Any]:
        """Extract features from items in parallel.
        
        Args:
            items: List of items to extract features from
            feature_func: Function to extract features from each item
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of extracted features
        """
        logger.info(f"Extracting features from {len(items)} items in parallel")
        
        try:
            with self.processor:
                if progress_callback:
                    results = self.processor.map_parallel_with_progress(
                        feature_func, items, progress_callback
                    )
                else:
                    results = self.processor.map_parallel(feature_func, items)
            
            successful = sum(1 for r in results if r is not None)
            failed = len(results) - successful
            
            logger.info(f"Parallel feature extraction completed: {successful} successful, {failed} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel feature extraction failed: {e}")
            raise

    def extract_embeddings_parallel(self, contents: List[str], 
                                   encoder_func: Callable,
                                   batch_size: int = 32) -> List[Any]:
        """Extract embeddings in parallel batches.
        
        Args:
            contents: List of content strings
            encoder_func: Function to encode content batches
            batch_size: Size of each encoding batch
            
        Returns:
            List of embeddings
        """
        logger.info(f"Extracting embeddings from {len(contents)} contents in parallel batches")
        
        try:
            # Create batches for efficient encoding
            batches = [contents[i:i + batch_size] for i in range(0, len(contents), batch_size)]
            
            # Process batches in parallel
            batch_processor = ParallelBatchProcessor(self.max_workers, batch_size)
            
            def process_embedding_batch(batch):
                try:
                    return encoder_func(batch)
                except Exception as e:
                    logger.warning(f"Embedding batch failed: {e}")
                    return [None] * len(batch)
            
            all_embeddings = batch_processor.process_batches(batches, process_embedding_batch)
            
            logger.info(f"Parallel embedding extraction completed: {len(all_embeddings)} embeddings")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Parallel embedding extraction failed: {e}")
            raise


def create_worker_pool(max_workers: Optional[int] = None, 
                      use_processes: bool = False) -> ParallelProcessor:
    """Create a configured worker pool.
    
    Args:
        max_workers: Maximum number of workers (None for auto)
        use_processes: Whether to use processes instead of threads
        
    Returns:
        Configured ParallelProcessor
    """
    return ParallelProcessor(max_workers, use_processes)


def parallel_map(func: Callable, items: List[Any], 
                max_workers: Optional[int] = None) -> List[Any]:
    """Simple parallel map function.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers (None for auto)
        
    Returns:
        List of results
    """
    with create_worker_pool(max_workers) as processor:
        return processor.map_parallel(func, items)