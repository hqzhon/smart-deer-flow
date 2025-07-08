#!/usr/bin/env python3
"""
DeerFlow Performance Test Script

This script is used to test DeerFlow's performance optimization features, including:
- Concurrent request testing
- Connection pool performance testing
- Cache effectiveness testing
- Error recovery testing
- Rate limiting testing

Usage:
    python scripts/performance_test.py --help
"""

import asyncio
import aiohttp
import argparse
import json
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data class"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    requests_per_second: float
    error_rate: float
    duration: float


class PerformanceTester:
    """DeerFlow performance tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=50,  # Connections per host
            ttl_dns_cache=300,  # DNS cache time
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check service health status"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Health check: {data.get('status', 'unknown')}")
                    return data.get('status') == 'healthy'
                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    async def single_request(self, message: str = "Test message") -> Dict[str, Any]:
        """Send a single chat request"""
        start_time = time.time()
        try:
            payload = {
                "message": message,
                "max_iterations": 2,
                "thread_id": f"test_{int(time.time() * 1000)}"
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/stream",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    # Read streaming response
                    chunks = []
                    async for line in response.content:
                        if line.strip():
                            try:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    data = json.loads(line_str[6:])
                                    chunks.append(data)
                            except json.JSONDecodeError:
                                continue
                    
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "chunks_count": len(chunks)
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e)
            }
    
    async def concurrent_test(
        self, 
        num_requests: int = 50, 
        concurrency: int = 10,
        message: str = "Concurrent test message"
    ) -> TestResult:
        """Concurrent request test"""
        logger.info(f"Starting concurrent test: {num_requests} requests, {concurrency} concurrency")
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request():
            async with semaphore:
                return await self.single_request(message)
        
        # Create tasks
        tasks = [limited_request() for _ in range(num_requests)]
        
        # Execute tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "response_time": 0})
            elif result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # Calculate statistics
        response_times = [r["response_time"] for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        requests_per_second = num_requests / duration if duration > 0 else 0
        error_rate = failed_count / num_requests if num_requests > 0 else 0
        
        return TestResult(
            test_name="Concurrent Request Test",
            total_requests=num_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            duration=duration
        )
    
    async def cache_test(self, num_requests: int = 20) -> TestResult:
        """Cache effectiveness test"""
        logger.info(f"Starting cache test: {num_requests} requests")
        
        # Use the same message to test cache effectiveness
        cache_message = "This is a cache test message that should be cached to improve response speed for subsequent requests."
        
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            result = await self.single_request(cache_message)
            results.append(result)
            
            # Record response time for each request
            if result.get("success"):
                logger.debug(f"Request {i+1}: {result['response_time']:.3f}s")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze cache effectiveness
        successful_results = [r for r in results if r.get("success")]
        response_times = [r["response_time"] for r in successful_results]
        
        if len(response_times) >= 2:
            # Compare first request with subsequent requests response time
            first_request_time = response_times[0]
            subsequent_times = response_times[1:]
            avg_subsequent_time = statistics.mean(subsequent_times)
            
            cache_improvement = (first_request_time - avg_subsequent_time) / first_request_time * 100
            logger.info(f"Cache improvement: {cache_improvement:.1f}%")
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        successful_count = len(successful_results)
        failed_count = len(results) - successful_count
        requests_per_second = num_requests / duration if duration > 0 else 0
        error_rate = failed_count / num_requests if num_requests > 0 else 0
        
        return TestResult(
            test_name="Cache Effectiveness Test",
            total_requests=num_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            duration=duration
        )
    
    async def stress_test(
        self, 
        duration_seconds: int = 60, 
        max_concurrency: int = 50
    ) -> TestResult:
        """Stress test"""
        logger.info(f"Starting stress test: {duration_seconds} seconds, max concurrency {max_concurrency}")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        results = []
        request_count = 0
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def stress_request():
            nonlocal request_count
            request_count += 1
            async with semaphore:
                return await self.single_request(f"Stress test message {request_count}")
        
        # Continuously send requests until time ends
        tasks = []
        while time.time() < end_time:
            task = asyncio.create_task(stress_request())
            tasks.append(task)
            
            # Control request frequency
            await asyncio.sleep(0.01)
        
        # Wait for all tasks to complete
        logger.info(f"Waiting for {len(tasks)} tasks to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        # Analyze results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "response_time": 0})
            elif result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # Calculate statistics
        response_times = [r["response_time"] for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        total_requests = len(results)
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        requests_per_second = total_requests / actual_duration if actual_duration > 0 else 0
        error_rate = failed_count / total_requests if total_requests > 0 else 0
        
        return TestResult(
            test_name="Stress Test",
            total_requests=total_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            duration=actual_duration
        )
    
    def print_result(self, result: TestResult):
        """Print test results"""
        print(f"\n{'='*60}")
        print(f"Test Name: {result.test_name}")
        print(f"{'='*60}")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful Requests: {result.successful_requests}")
        print(f"Failed Requests: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate:.2%}")
        print(f"Test Duration: {result.duration:.2f} seconds")
        print(f"Requests/Second: {result.requests_per_second:.2f}")
        print(f"")
        print(f"Response Time Statistics:")
        print(f"  Average: {result.avg_response_time:.3f} seconds")
        print(f"  Minimum: {result.min_response_time:.3f} seconds")
        print(f"  Maximum: {result.max_response_time:.3f} seconds")
        print(f"  P95: {result.p95_response_time:.3f} seconds")
        print(f"{'='*60}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DeerFlow performance testing tool")
    parser.add_argument("--url", default="http://localhost:8000", help="DeerFlow service URL")
    parser.add_argument("--test", choices=["all", "concurrent", "cache", "stress"], 
                       default="all", help="Test type")
    parser.add_argument("--requests", type=int, default=50, help="Number of concurrent test requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--duration", type=int, default=60, help="Stress test duration (seconds)")
    parser.add_argument("--cache-requests", type=int, default=20, help="Number of cache test requests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async with PerformanceTester(args.url) as tester:
        # Health check
        logger.info("Performing health check...")
        if not await tester.health_check():
            logger.error("Service health check failed, please ensure DeerFlow service is running")
            return
        
        # Get initial metrics
        logger.info("Getting initial performance metrics...")
        initial_metrics = await tester.get_metrics()
        if initial_metrics:
            logger.info(f"Connection pool utilization: {initial_metrics.get('connection_pool', {}).get('utilization', 0):.1%}")
            logger.info(f"Request queue size: {initial_metrics.get('request_queue', {}).get('size', 0)}")
        
        results = []
        
        # Execute tests
        if args.test in ["all", "concurrent"]:
            logger.info("\nStarting concurrent test...")
            concurrent_result = await tester.concurrent_test(
                num_requests=args.requests,
                concurrency=args.concurrency
            )
            results.append(concurrent_result)
            tester.print_result(concurrent_result)
        
        if args.test in ["all", "cache"]:
            logger.info("\nStarting cache test...")
            cache_result = await tester.cache_test(num_requests=args.cache_requests)
            results.append(cache_result)
            tester.print_result(cache_result)
        
        if args.test in ["all", "stress"]:
            logger.info("\nStarting stress test...")
            stress_result = await tester.stress_test(
                duration_seconds=args.duration,
                max_concurrency=args.concurrency * 2
            )
            results.append(stress_result)
            tester.print_result(stress_result)
        
        # Get final metrics
        logger.info("\nGetting final performance metrics...")
        final_metrics = await tester.get_metrics()
        if final_metrics:
            print(f"\nFinal system status:")
            print(f"Connection pool utilization: {final_metrics.get('connection_pool', {}).get('utilization', 0):.1%}")
            print(f"Request queue size: {final_metrics.get('request_queue', {}).get('size', 0)}")
            
            conn_metrics = final_metrics.get('connection_pool', {})
            if conn_metrics:
                print(f"Total connections acquired: {conn_metrics.get('total_acquired', 0)}")
                print(f"Total connections released: {conn_metrics.get('total_released', 0)}")
                print(f"Peak connection usage: {conn_metrics.get('peak_usage', 0)}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Test Summary")
        print(f"{'='*60}")
        for result in results:
            print(f"{result.test_name}: {result.requests_per_second:.2f} RPS, "
                  f"Error rate {result.error_rate:.2%}, "
                  f"Average response time {result.avg_response_time:.3f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")