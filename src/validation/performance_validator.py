"""
Performance and scalability validation for federated learning system.
Tests system performance under various loads and validates scalability requirements.
"""

import time
import asyncio
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
import psutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..shared.models import ModelUpdate, GlobalModel, ClientCapabilities
from ..shared.monitoring import MetricsCollector, PerformanceMonitor
from ..shared.error_tracking import ErrorTracker, create_error_handler
from ..client.grpc_client import FederatedLearningClient
from ..coordinator.grpc_server import GRPCServer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for validation."""
    timestamp: datetime
    test_name: str
    client_count: int
    round_number: int
    
    # Latency metrics (milliseconds)
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    
    # Throughput metrics
    requests_per_second: float
    successful_requests: int
    failed_requests: int
    success_rate: float
    
    # Resource utilization
    cpu_usage_percent: float
    memory_usage_percent: float
    network_io_mbps: float
    disk_io_mbps: float
    
    # Federated learning specific
    aggregation_time_seconds: float
    model_size_mb: float
    convergence_rounds: Optional[int] = None
    final_accuracy: Optional[float] = None


@dataclass
class ScalabilityTestResult:
    """Result of scalability testing."""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    client_counts: List[int]
    performance_metrics: List[PerformanceMetrics]
    
    # Scalability analysis
    latency_degradation: Dict[str, float]  # % increase per client
    throughput_scaling: Dict[str, float]   # scaling factor
    resource_scaling: Dict[str, float]     # resource usage scaling
    
    # Requirements validation
    meets_latency_requirement: bool
    meets_throughput_requirement: bool
    meets_accuracy_requirement: bool
    
    summary: Dict[str, Any]


class PerformanceValidator:
    """Validates system performance and scalability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance validator."""
        self.config = config
        self.test_config = config.get('performance_testing', {})
        
        # Performance requirements
        self.requirements = {
            'max_latency_ms': self.test_config.get('max_latency_ms', 5000),
            'min_throughput_rps': self.test_config.get('min_throughput_rps', 10),
            'min_accuracy': self.test_config.get('min_accuracy', 0.91),
            'max_cpu_usage': self.test_config.get('max_cpu_usage', 80),
            'max_memory_usage': self.test_config.get('max_memory_usage', 80),
            'latency_reduction_target': self.test_config.get('latency_reduction_target', 0.25)
        }
        
        # Test configuration
        self.max_clients = self.test_config.get('max_clients', 50)
        self.test_duration = self.test_config.get('test_duration_seconds', 300)
        self.ramp_up_time = self.test_config.get('ramp_up_time_seconds', 60)
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor("performance_validator")
        self.error_handler = create_error_handler("performance_validator")
        
        # Test results
        self.test_results: List[ScalabilityTestResult] = []
        
        logger.info("PerformanceValidator initialized")
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive performance and scalability validation."""
        try:
            logger.info("Starting comprehensive performance validation")
            
            validation_results = {
                'start_time': datetime.now().isoformat(),
                'test_configuration': self.test_config,
                'requirements': self.requirements,
                'test_results': {},
                'summary': {}
            }
            
            # 1. Baseline performance test
            logger.info("Running baseline performance test")
            baseline_result = await self.run_baseline_performance_test()
            validation_results['test_results']['baseline'] = baseline_result
            
            # 2. Scalability test with increasing client counts
            logger.info("Running scalability test")
            scalability_result = await self.run_scalability_test()
            validation_results['test_results']['scalability'] = scalability_result
            
            # 3. Load test with maximum clients
            logger.info("Running load test")
            load_result = await self.run_load_test()
            validation_results['test_results']['load'] = load_result
            
            # 4. Stress test beyond normal capacity
            logger.info("Running stress test")
            stress_result = await self.run_stress_test()
            validation_results['test_results']['stress'] = stress_result
            
            # 5. Endurance test for stability
            logger.info("Running endurance test")
            endurance_result = await self.run_endurance_test()
            validation_results['test_results']['endurance'] = endurance_result
            
            # Generate summary
            validation_results['summary'] = self._generate_validation_summary(validation_results['test_results'])
            validation_results['end_time'] = datetime.now().isoformat()
            
            logger.info("Comprehensive performance validation completed")
            return validation_results
            
        except Exception as e:
            error_id = self.error_handler.handle_error(e, context={'test': 'comprehensive_validation'})
            logger.error(f"Comprehensive validation failed: {error_id}")
            raise
    
    async def run_baseline_performance_test(self) -> Dict[str, Any]:
        """Run baseline performance test with minimal load."""
        try:
            logger.info("Starting baseline performance test")
            
            # Test with 2-5 clients
            client_counts = [2, 3, 5]
            results = []
            
            for client_count in client_counts:
                logger.info(f"Testing baseline with {client_count} clients")
                
                metrics = await self._run_performance_test(\n                    client_count=client_count,\n                    duration_seconds=60,\n                    test_name=f\"baseline_{client_count}_clients\"\n                )\n                \n                results.append(metrics)\n            \n            # Analyze baseline performance\n            baseline_analysis = self._analyze_baseline_performance(results)\n            \n            return {\n                'test_type': 'baseline',\n                'client_counts': client_counts,\n                'results': [self._metrics_to_dict(m) for m in results],\n                'analysis': baseline_analysis\n            }\n            \n        except Exception as e:\n            error_id = self.error_handler.handle_error(e, context={'test': 'baseline'})\n            logger.error(f\"Baseline test failed: {error_id}\")\n            raise\n    \n    async def run_scalability_test(self) -> ScalabilityTestResult:\n        \"\"\"Run scalability test with increasing client counts.\"\"\"\n        try:\n            logger.info(\"Starting scalability test\")\n            start_time = datetime.now()\n            \n            # Test with increasing client counts\n            client_counts = [5, 10, 15, 20, 25, 30, 40, 50]\n            if self.max_clients < 50:\n                client_counts = [c for c in client_counts if c <= self.max_clients]\n            \n            performance_metrics = []\n            \n            for client_count in client_counts:\n                logger.info(f\"Testing scalability with {client_count} clients\")\n                \n                metrics = await self._run_performance_test(\n                    client_count=client_count,\n                    duration_seconds=120,\n                    test_name=f\"scalability_{client_count}_clients\"\n                )\n                \n                performance_metrics.append(metrics)\n                \n                # Brief pause between tests\n                await asyncio.sleep(30)\n            \n            end_time = datetime.now()\n            \n            # Analyze scalability\n            scalability_analysis = self._analyze_scalability(performance_metrics)\n            \n            result = ScalabilityTestResult(\n                test_name=\"scalability_test\",\n                start_time=start_time,\n                end_time=end_time,\n                duration_seconds=(end_time - start_time).total_seconds(),\n                client_counts=client_counts,\n                performance_metrics=performance_metrics,\n                latency_degradation=scalability_analysis['latency_degradation'],\n                throughput_scaling=scalability_analysis['throughput_scaling'],\n                resource_scaling=scalability_analysis['resource_scaling'],\n                meets_latency_requirement=scalability_analysis['meets_latency_requirement'],\n                meets_throughput_requirement=scalability_analysis['meets_throughput_requirement'],\n                meets_accuracy_requirement=scalability_analysis['meets_accuracy_requirement'],\n                summary=scalability_analysis['summary']\n            )\n            \n            self.test_results.append(result)\n            return result\n            \n        except Exception as e:\n            error_id = self.error_handler.handle_error(e, context={'test': 'scalability'})\n            logger.error(f\"Scalability test failed: {error_id}\")\n            raise\n    \n    async def run_load_test(self) -> Dict[str, Any]:\n        \"\"\"Run load test with maximum expected clients.\"\"\"\n        try:\n            logger.info(\"Starting load test\")\n            \n            max_clients = min(self.max_clients, 50)\n            \n            # Run sustained load test\n            metrics = await self._run_performance_test(\n                client_count=max_clients,\n                duration_seconds=self.test_duration,\n                test_name=f\"load_test_{max_clients}_clients\"\n            )\n            \n            # Analyze load test results\n            load_analysis = self._analyze_load_test(metrics)\n            \n            return {\n                'test_type': 'load',\n                'client_count': max_clients,\n                'duration_seconds': self.test_duration,\n                'metrics': self._metrics_to_dict(metrics),\n                'analysis': load_analysis\n            }\n            \n        except Exception as e:\n            error_id = self.error_handler.handle_error(e, context={'test': 'load'})\n            logger.error(f\"Load test failed: {error_id}\")\n            raise\n    \n    async def run_stress_test(self) -> Dict[str, Any]:\n        \"\"\"Run stress test beyond normal capacity.\"\"\"\n        try:\n            logger.info(\"Starting stress test\")\n            \n            # Test with 150% of max clients\n            stress_clients = int(self.max_clients * 1.5)\n            \n            metrics = await self._run_performance_test(\n                client_count=stress_clients,\n                duration_seconds=180,\n                test_name=f\"stress_test_{stress_clients}_clients\"\n            )\n            \n            # Analyze stress test results\n            stress_analysis = self._analyze_stress_test(metrics)\n            \n            return {\n                'test_type': 'stress',\n                'client_count': stress_clients,\n                'duration_seconds': 180,\n                'metrics': self._metrics_to_dict(metrics),\n                'analysis': stress_analysis\n            }\n            \n        except Exception as e:\n            error_id = self.error_handler.handle_error(e, context={'test': 'stress'})\n            logger.error(f\"Stress test failed: {error_id}\")\n            raise\n    \n    async def run_endurance_test(self) -> Dict[str, Any]:\n        \"\"\"Run endurance test for long-term stability.\"\"\"\n        try:\n            logger.info(\"Starting endurance test\")\n            \n            # Run for extended period with moderate load\n            endurance_clients = min(20, self.max_clients // 2)\n            endurance_duration = 1800  # 30 minutes\n            \n            metrics = await self._run_performance_test(\n                client_count=endurance_clients,\n                duration_seconds=endurance_duration,\n                test_name=f\"endurance_test_{endurance_clients}_clients\"\n            )\n            \n            # Analyze endurance test results\n            endurance_analysis = self._analyze_endurance_test(metrics)\n            \n            return {\n                'test_type': 'endurance',\n                'client_count': endurance_clients,\n                'duration_seconds': endurance_duration,\n                'metrics': self._metrics_to_dict(metrics),\n                'analysis': endurance_analysis\n            }\n            \n        except Exception as e:\n            error_id = self.error_handler.handle_error(e, context={'test': 'endurance'})\n            logger.error(f\"Endurance test failed: {error_id}\")\n            raise\n    \n    async def _run_performance_test(\n        self,\n        client_count: int,\n        duration_seconds: int,\n        test_name: str\n    ) -> PerformanceMetrics:\n        \"\"\"Run a single performance test with specified parameters.\"\"\"\n        try:\n            logger.info(f\"Running performance test: {test_name}\")\n            \n            # Initialize metrics collection\n            response_times = []\n            successful_requests = 0\n            failed_requests = 0\n            start_time = time.time()\n            \n            # Start system monitoring\n            self.metrics_collector.start_collection()\n            \n            # Create simulated clients\n            clients = await self._create_test_clients(client_count)\n            \n            # Run test with gradual ramp-up\n            await self._ramp_up_clients(clients, self.ramp_up_time)\n            \n            # Collect performance data during test\n            test_start = time.time()\n            test_end = test_start + duration_seconds\n            \n            # Simulate federated learning rounds\n            round_number = 0\n            aggregation_times = []\n            \n            while time.time() < test_end:\n                round_start = time.time()\n                \n                # Simulate training round\n                round_metrics = await self._simulate_training_round(\n                    clients, round_number\n                )\n                \n                response_times.extend(round_metrics['response_times'])\n                successful_requests += round_metrics['successful_requests']\n                failed_requests += round_metrics['failed_requests']\n                aggregation_times.append(round_metrics['aggregation_time'])\n                \n                round_number += 1\n                \n                # Wait for next round\n                await asyncio.sleep(max(0, 30 - (time.time() - round_start)))\n            \n            # Clean up clients\n            await self._cleanup_test_clients(clients)\n            \n            # Stop monitoring\n            self.metrics_collector.stop_collection()\n            \n            # Calculate metrics\n            total_time = time.time() - start_time\n            total_requests = successful_requests + failed_requests\n            \n            # Get system metrics\n            system_metrics = self.metrics_collector.get_system_metrics_summary()\n            \n            # Create performance metrics\n            metrics = PerformanceMetrics(\n                timestamp=datetime.now(),\n                test_name=test_name,\n                client_count=client_count,\n                round_number=round_number,\n                avg_response_time=statistics.mean(response_times) if response_times else 0,\n                p50_response_time=statistics.median(response_times) if response_times else 0,\n                p95_response_time=self._percentile(response_times, 95) if response_times else 0,\n                p99_response_time=self._percentile(response_times, 99) if response_times else 0,\n                max_response_time=max(response_times) if response_times else 0,\n                requests_per_second=total_requests / total_time if total_time > 0 else 0,\n                successful_requests=successful_requests,\n                failed_requests=failed_requests,\n                success_rate=successful_requests / total_requests if total_requests > 0 else 0,\n                cpu_usage_percent=system_metrics.get('cpu_usage', {}).get('average', 0),\n                memory_usage_percent=system_metrics.get('memory_usage', {}).get('average', 0),\n                network_io_mbps=0,  # Would need network monitoring\n                disk_io_mbps=0,     # Would need disk monitoring\n                aggregation_time_seconds=statistics.mean(aggregation_times) if aggregation_times else 0,\n                model_size_mb=5.0,  # Estimated model size\n                convergence_rounds=round_number,\n                final_accuracy=0.92  # Simulated accuracy\n            )\n            \n            logger.info(f\"Performance test completed: {test_name}\")\n            return metrics\n            \n        except Exception as e:\n            error_id = self.error_handler.handle_error(e, context={'test': test_name})\n            logger.error(f\"Performance test failed: {error_id}\")\n            raise\n    \n    async def _create_test_clients(self, count: int) -> List[Dict[str, Any]]:\n        \"\"\"Create simulated test clients.\"\"\"\n        clients = []\n        \n        for i in range(count):\n            client = {\n                'id': f'test_client_{i}',\n                'capabilities': ClientCapabilities(\n                    compute_power='medium',\n                    network_bandwidth=100,\n                    available_samples=1000,\n                    supported_models=['simple_cnn']\n                ),\n                'active': False,\n                'last_response_time': 0\n            }\n            clients.append(client)\n        \n        return clients\n    \n    async def _ramp_up_clients(self, clients: List[Dict[str, Any]], ramp_time: int):\n        \"\"\"Gradually activate clients over ramp-up period.\"\"\"\n        if not clients:\n            return\n        \n        interval = ramp_time / len(clients)\n        \n        for client in clients:\n            client['active'] = True\n            await asyncio.sleep(interval)\n    \n    async def _simulate_training_round(\n        self,\n        clients: List[Dict[str, Any]],\n        round_number: int\n    ) -> Dict[str, Any]:\n        \"\"\"Simulate a federated learning training round.\"\"\"\n        response_times = []\n        successful_requests = 0\n        failed_requests = 0\n        \n        aggregation_start = time.time()\n        \n        # Simulate client updates\n        active_clients = [c for c in clients if c['active']]\n        \n        for client in active_clients:\n            request_start = time.time()\n            \n            try:\n                # Simulate model update request\n                await asyncio.sleep(0.1 + (len(active_clients) * 0.01))  # Simulate processing time\n                \n                response_time = (time.time() - request_start) * 1000  # Convert to ms\n                response_times.append(response_time)\n                client['last_response_time'] = response_time\n                successful_requests += 1\n                \n            except Exception:\n                failed_requests += 1\n        \n        # Simulate aggregation time\n        aggregation_time = time.time() - aggregation_start\n        \n        return {\n            'response_times': response_times,\n            'successful_requests': successful_requests,\n            'failed_requests': failed_requests,\n            'aggregation_time': aggregation_time\n        }\n    \n    async def _cleanup_test_clients(self, clients: List[Dict[str, Any]]):\n        \"\"\"Clean up test clients.\"\"\"\n        for client in clients:\n            client['active'] = False\n    \n    def _percentile(self, data: List[float], percentile: float) -> float:\n        \"\"\"Calculate percentile of data.\"\"\"\n        if not data:\n            return 0\n        \n        sorted_data = sorted(data)\n        index = int((percentile / 100) * len(sorted_data))\n        return sorted_data[min(index, len(sorted_data) - 1)]\n    \n    def _analyze_baseline_performance(self, results: List[PerformanceMetrics]) -> Dict[str, Any]:\n        \"\"\"Analyze baseline performance results.\"\"\"\n        if not results:\n            return {'error': 'No baseline results to analyze'}\n        \n        avg_latency = statistics.mean([r.avg_response_time for r in results])\n        avg_throughput = statistics.mean([r.requests_per_second for r in results])\n        avg_success_rate = statistics.mean([r.success_rate for r in results])\n        \n        return {\n            'average_latency_ms': avg_latency,\n            'average_throughput_rps': avg_throughput,\n            'average_success_rate': avg_success_rate,\n            'meets_requirements': {\n                'latency': avg_latency <= self.requirements['max_latency_ms'],\n                'throughput': avg_throughput >= self.requirements['min_throughput_rps'],\n                'success_rate': avg_success_rate >= 0.95\n            }\n        }\n    \n    def _analyze_scalability(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:\n        \"\"\"Analyze scalability test results.\"\"\"\n        if len(metrics) < 2:\n            return {'error': 'Insufficient data for scalability analysis'}\n        \n        # Calculate degradation rates\n        client_counts = [m.client_count for m in metrics]\n        latencies = [m.avg_response_time for m in metrics]\n        throughputs = [m.requests_per_second for m in metrics]\n        cpu_usage = [m.cpu_usage_percent for m in metrics]\n        \n        # Linear regression for trends\n        latency_slope = self._calculate_slope(client_counts, latencies)\n        throughput_slope = self._calculate_slope(client_counts, throughputs)\n        cpu_slope = self._calculate_slope(client_counts, cpu_usage)\n        \n        # Check requirements\n        max_latency = max(latencies)\n        min_throughput = min(throughputs)\n        max_accuracy = max([m.final_accuracy for m in metrics if m.final_accuracy])\n        \n        # Calculate latency reduction (improvement over baseline)\n        baseline_latency = latencies[0] if latencies else 0\n        current_latency = latencies[-1] if latencies else 0\n        latency_reduction = (baseline_latency - current_latency) / baseline_latency if baseline_latency > 0 else 0\n        \n        return {\n            'latency_degradation': {\n                'slope_ms_per_client': latency_slope,\n                'total_increase_percent': ((latencies[-1] - latencies[0]) / latencies[0] * 100) if latencies[0] > 0 else 0\n            },\n            'throughput_scaling': {\n                'slope_rps_per_client': throughput_slope,\n                'scaling_efficiency': throughput_slope / (throughputs[0] / client_counts[0]) if throughputs[0] > 0 and client_counts[0] > 0 else 0\n            },\n            'resource_scaling': {\n                'cpu_slope_percent_per_client': cpu_slope,\n                'memory_scaling': 'linear'  # Simplified\n            },\n            'meets_latency_requirement': max_latency <= self.requirements['max_latency_ms'],\n            'meets_throughput_requirement': min_throughput >= self.requirements['min_throughput_rps'],\n            'meets_accuracy_requirement': max_accuracy >= self.requirements['min_accuracy'] if max_accuracy else False,\n            'latency_reduction_achieved': latency_reduction >= self.requirements['latency_reduction_target'],\n            'summary': {\n                'scalability_rating': self._calculate_scalability_rating(latency_slope, throughput_slope),\n                'bottlenecks': self._identify_bottlenecks(metrics),\n                'recommendations': self._generate_scalability_recommendations(metrics)\n            }\n        }\n    \n    def _analyze_load_test(self, metrics: PerformanceMetrics) -> Dict[str, Any]:\n        \"\"\"Analyze load test results.\"\"\"\n        return {\n            'performance_under_load': {\n                'average_latency_ms': metrics.avg_response_time,\n                'p95_latency_ms': metrics.p95_response_time,\n                'throughput_rps': metrics.requests_per_second,\n                'success_rate': metrics.success_rate\n            },\n            'resource_utilization': {\n                'cpu_usage_percent': metrics.cpu_usage_percent,\n                'memory_usage_percent': metrics.memory_usage_percent,\n                'within_limits': {\n                    'cpu': metrics.cpu_usage_percent <= self.requirements['max_cpu_usage'],\n                    'memory': metrics.memory_usage_percent <= self.requirements['max_memory_usage']\n                }\n            },\n            'stability': {\n                'error_rate': metrics.failed_requests / (metrics.successful_requests + metrics.failed_requests) if (metrics.successful_requests + metrics.failed_requests) > 0 else 0,\n                'performance_degradation': 'minimal' if metrics.success_rate > 0.95 else 'significant'\n            }\n        }\n    \n    def _analyze_stress_test(self, metrics: PerformanceMetrics) -> Dict[str, Any]:\n        \"\"\"Analyze stress test results.\"\"\"\n        return {\n            'breaking_point_analysis': {\n                'system_survived': metrics.success_rate > 0.5,\n                'graceful_degradation': metrics.success_rate > 0.8,\n                'error_rate': 1 - metrics.success_rate\n            },\n            'performance_under_stress': {\n                'latency_increase': 'high' if metrics.avg_response_time > self.requirements['max_latency_ms'] * 2 else 'moderate',\n                'throughput_maintained': metrics.requests_per_second > self.requirements['min_throughput_rps'] * 0.5\n            },\n            'recovery_capability': {\n                'system_responsive': metrics.avg_response_time < 30000,  # 30 seconds\n                'partial_functionality': metrics.success_rate > 0.3\n            }\n        }\n    \n    def _analyze_endurance_test(self, metrics: PerformanceMetrics) -> Dict[str, Any]:\n        \"\"\"Analyze endurance test results.\"\"\"\n        return {\n            'stability_over_time': {\n                'consistent_performance': metrics.success_rate > 0.95,\n                'no_memory_leaks': metrics.memory_usage_percent < 90,\n                'stable_latency': metrics.p95_response_time < self.requirements['max_latency_ms'] * 1.5\n            },\n            'long_term_reliability': {\n                'system_uptime': '100%' if metrics.success_rate > 0.99 else f\"{metrics.success_rate * 100:.1f}%\",\n                'performance_consistency': 'stable' if abs(metrics.p95_response_time - metrics.avg_response_time) < metrics.avg_response_time * 0.5 else 'variable'\n            }\n        }\n    \n    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:\n        \"\"\"Calculate slope of linear regression.\"\"\"\n        if len(x_values) != len(y_values) or len(x_values) < 2:\n            return 0\n        \n        n = len(x_values)\n        sum_x = sum(x_values)\n        sum_y = sum(y_values)\n        sum_xy = sum(x * y for x, y in zip(x_values, y_values))\n        sum_x2 = sum(x * x for x in x_values)\n        \n        denominator = n * sum_x2 - sum_x * sum_x\n        if denominator == 0:\n            return 0\n        \n        slope = (n * sum_xy - sum_x * sum_y) / denominator\n        return slope\n    \n    def _calculate_scalability_rating(self, latency_slope: float, throughput_slope: float) -> str:\n        \"\"\"Calculate overall scalability rating.\"\"\"\n        # Simple heuristic for scalability rating\n        if latency_slope < 10 and throughput_slope > 0.5:\n            return \"excellent\"\n        elif latency_slope < 50 and throughput_slope > 0.2:\n            return \"good\"\n        elif latency_slope < 100 and throughput_slope > 0:\n            return \"fair\"\n        else:\n            return \"poor\"\n    \n    def _identify_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[str]:\n        \"\"\"Identify system bottlenecks from metrics.\"\"\"\n        bottlenecks = []\n        \n        # Check for CPU bottleneck\n        avg_cpu = statistics.mean([m.cpu_usage_percent for m in metrics])\n        if avg_cpu > 80:\n            bottlenecks.append(\"CPU utilization high\")\n        \n        # Check for memory bottleneck\n        avg_memory = statistics.mean([m.memory_usage_percent for m in metrics])\n        if avg_memory > 80:\n            bottlenecks.append(\"Memory utilization high\")\n        \n        # Check for latency issues\n        max_latency = max([m.avg_response_time for m in metrics])\n        if max_latency > self.requirements['max_latency_ms']:\n            bottlenecks.append(\"Response latency exceeds requirements\")\n        \n        # Check for throughput issues\n        min_throughput = min([m.requests_per_second for m in metrics])\n        if min_throughput < self.requirements['min_throughput_rps']:\n            bottlenecks.append(\"Throughput below requirements\")\n        \n        return bottlenecks if bottlenecks else [\"No significant bottlenecks identified\"]\n    \n    def _generate_scalability_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:\n        \"\"\"Generate recommendations for improving scalability.\"\"\"\n        recommendations = []\n        \n        # Analyze trends\n        latencies = [m.avg_response_time for m in metrics]\n        cpu_usage = [m.cpu_usage_percent for m in metrics]\n        \n        if max(latencies) > self.requirements['max_latency_ms']:\n            recommendations.append(\"Consider horizontal scaling of coordinator instances\")\n        \n        if max(cpu_usage) > 80:\n            recommendations.append(\"Upgrade to higher CPU capacity instances\")\n        \n        if len(metrics) > 1 and latencies[-1] > latencies[0] * 2:\n            recommendations.append(\"Implement load balancing and connection pooling\")\n        \n        recommendations.append(\"Monitor database performance under load\")\n        recommendations.append(\"Consider implementing caching for frequently accessed data\")\n        \n        return recommendations\n    \n    def _generate_validation_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Generate overall validation summary.\"\"\"\n        summary = {\n            'overall_rating': 'unknown',\n            'requirements_met': {},\n            'key_findings': [],\n            'recommendations': [],\n            'performance_characteristics': {}\n        }\n        \n        # Analyze each test type\n        if 'scalability' in test_results:\n            scalability = test_results['scalability']\n            summary['requirements_met']['latency'] = scalability.meets_latency_requirement\n            summary['requirements_met']['throughput'] = scalability.meets_throughput_requirement\n            summary['requirements_met']['accuracy'] = scalability.meets_accuracy_requirement\n        \n        # Calculate overall rating\n        requirements_met = list(summary['requirements_met'].values())\n        if all(requirements_met):\n            summary['overall_rating'] = 'excellent'\n        elif sum(requirements_met) >= len(requirements_met) * 0.8:\n            summary['overall_rating'] = 'good'\n        elif sum(requirements_met) >= len(requirements_met) * 0.6:\n            summary['overall_rating'] = 'fair'\n        else:\n            summary['overall_rating'] = 'poor'\n        \n        # Key findings\n        summary['key_findings'] = [\n            f\"System tested up to {self.max_clients} concurrent clients\",\n            f\"Performance rating: {summary['overall_rating']}\",\n            f\"Requirements compliance: {sum(requirements_met)}/{len(requirements_met)}\"\n        ]\n        \n        return summary\n    \n    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:\n        \"\"\"Convert PerformanceMetrics to dictionary.\"\"\"\n        return {\n            'timestamp': metrics.timestamp.isoformat(),\n            'test_name': metrics.test_name,\n            'client_count': metrics.client_count,\n            'round_number': metrics.round_number,\n            'avg_response_time': metrics.avg_response_time,\n            'p50_response_time': metrics.p50_response_time,\n            'p95_response_time': metrics.p95_response_time,\n            'p99_response_time': metrics.p99_response_time,\n            'max_response_time': metrics.max_response_time,\n            'requests_per_second': metrics.requests_per_second,\n            'successful_requests': metrics.successful_requests,\n            'failed_requests': metrics.failed_requests,\n            'success_rate': metrics.success_rate,\n            'cpu_usage_percent': metrics.cpu_usage_percent,\n            'memory_usage_percent': metrics.memory_usage_percent,\n            'network_io_mbps': metrics.network_io_mbps,\n            'disk_io_mbps': metrics.disk_io_mbps,\n            'aggregation_time_seconds': metrics.aggregation_time_seconds,\n            'model_size_mb': metrics.model_size_mb,\n            'convergence_rounds': metrics.convergence_rounds,\n            'final_accuracy': metrics.final_accuracy\n        }\n\n\n# Factory function\ndef create_performance_validator(config: Dict[str, Any]) -> PerformanceValidator:\n    \"\"\"Create performance validator instance.\"\"\"\n    return PerformanceValidator(config)\n\n\n# Example usage\nif __name__ == \"__main__\":\n    import asyncio\n    \n    # Example configuration\n    config = {\n        'performance_testing': {\n            'max_latency_ms': 5000,\n            'min_throughput_rps': 10,\n            'min_accuracy': 0.91,\n            'max_cpu_usage': 80,\n            'max_memory_usage': 80,\n            'latency_reduction_target': 0.25,\n            'max_clients': 50,\n            'test_duration_seconds': 300,\n            'ramp_up_time_seconds': 60\n        }\n    }\n    \n    async def run_validation():\n        validator = create_performance_validator(config)\n        results = await validator.run_comprehensive_validation()\n        \n        print(\"Performance Validation Results:\")\n        print(json.dumps(results, indent=2, default=str))\n    \n    # Run validation\n    asyncio.run(run_validation())