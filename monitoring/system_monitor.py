# monitoring/system_monitor.py - System Health Monitoring

import psutil
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import requests
import subprocess
import sys
import os

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DOWN = "DOWN"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_sent: int
    network_recv: int
    load_average: float
    uptime_seconds: float
    process_count: int
    thread_count: int

@dataclass
class ServiceHealth:
    """Individual service health status"""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    error_message: str = ""
    uptime_percentage: float = 100.0

@dataclass
class DatabaseHealth:
    """Database connection health"""
    database_name: str
    status: HealthStatus
    connection_time_ms: float
    query_time_ms: float
    last_check: datetime
    error_message: str = ""
    table_count: int = 0
    record_count: int = 0

@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True

class SystemMonitor:
    """
    Comprehensive System Health Monitoring
    
    Monitors:
    - System resources (CPU, Memory, Disk, Network)
    - Service availability and response times
    - Database connectivity and performance
    - Application-specific metrics
    - Process health and performance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = []
        self.service_health = {}
        self.database_health = {}
        self.alert_thresholds = {}
        
        # Configuration
        self.check_interval = self.config.get('check_interval', 30)  # seconds
        self.history_retention = self.config.get('history_retention', 24)  # hours
        self.max_history_records = self.config.get('max_history_records', 2880)  # 24h at 30s intervals
        
        # Initialize alert thresholds
        self._initialize_alert_thresholds()
        
        # Services to monitor
        self.services_to_monitor = self.config.get('services', [
            {'name': 'trading_api', 'url': 'http://localhost:8000/health', 'timeout': 5},
            {'name': 'market_data', 'check_type': 'process', 'process_name': 'market_data'},
            {'name': 'scheduler', 'check_type': 'process', 'process_name': 'scheduler'}
        ])
        
        # Databases to monitor
        self.databases_to_monitor = self.config.get('databases', [
            {'name': 'trading_db', 'path': 'trading_data.db', 'type': 'sqlite'}
        ])
        
        logger.info("System Monitor initialized")
    
    def _initialize_alert_thresholds(self):
        """Initialize default alert thresholds"""
        default_thresholds = [
            AlertThreshold("cpu_percent", 80.0, 95.0),
            AlertThreshold("memory_percent", 85.0, 95.0),
            AlertThreshold("disk_percent", 90.0, 98.0),
            AlertThreshold("load_average", 2.0, 4.0),
            AlertThreshold("response_time_ms", 1000.0, 5000.0),
            AlertThreshold("database_query_time_ms", 500.0, 2000.0),
        ]
        
        for threshold in default_thresholds:
            self.alert_thresholds[threshold.metric_name] = threshold
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.is_monitoring:
            logger.warning("System monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check service health
                self._check_services_health()
                
                # Check database health
                self._check_database_health()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Check alert conditions
                self._check_alert_conditions(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_recv = network.bytes_recv
            
            # Load average (Unix-like systems)
            try:
                load_average = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            except:
                load_average = 0.0
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            # Process information
            process_count = len(psutil.pids())
            
            # Thread count for current process
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_sent=network_sent,
                network_recv=network_recv,
                load_average=load_average,
                uptime_seconds=uptime_seconds,
                process_count=process_count,
                thread_count=thread_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            # Return empty metrics with current timestamp
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0, memory_percent=0.0, disk_percent=0.0,
                network_sent=0, network_recv=0, load_average=0.0,
                uptime_seconds=0.0, process_count=0, thread_count=0
            )
    
    def _check_services_health(self):
        """Check health of configured services"""
        for service_config in self.services_to_monitor:
            service_name = service_config['name']
            
            try:
                if service_config.get('check_type') == 'process':
                    health = self._check_process_health(service_config)
                else:
                    health = self._check_url_health(service_config)
                
                self.service_health[service_name] = health
                
            except Exception as e:
                logger.error(f"Error checking service {service_name}: {str(e)}")
                self.service_health[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.CRITICAL,
                    response_time_ms=0.0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
    
    def _check_url_health(self, service_config: Dict[str, Any]) -> ServiceHealth:
        """Check health of URL-based service"""
        service_name = service_config['name']
        url = service_config['url']
        timeout = service_config.get('timeout', 5)
        
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=timeout)
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                error_message = ""
            else:
                status = HealthStatus.WARNING
                error_message = f"HTTP {response.status_code}"
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                error_message=error_message
            )
            
        except requests.exceptions.Timeout:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                error_message="Request timeout"
            )
        except requests.exceptions.ConnectionError:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.DOWN,
                response_time_ms=0.0,
                last_check=datetime.now(),
                error_message="Connection failed"
            )
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    def _check_process_health(self, service_config: Dict[str, Any]) -> ServiceHealth:
        """Check health of process-based service"""
        service_name = service_config['name']
        process_name = service_config.get('process_name', service_name)
        
        try:
            # Find processes by name
            matching_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        matching_processes.append(proc)
                    elif proc.info['cmdline'] and any(process_name.lower() in arg.lower() for arg in proc.info['cmdline']):
                        matching_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if matching_processes:
                # Process found and running
                proc = matching_processes[0]  # Use first match
                
                # Check process health metrics
                try:
                    cpu_percent = proc.cpu_percent()
                    memory_percent = proc.memory_percent()
                    
                    # Determine health based on resource usage
                    if cpu_percent > 90 or memory_percent > 90:
                        status = HealthStatus.WARNING
                        error_message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
                    else:
                        status = HealthStatus.HEALTHY
                        error_message = ""
                    
                    # Response time is process age (lower is better for recently restarted processes)
                    create_time = proc.create_time()
                    response_time_ms = (time.time() - create_time) / 3600 * 1000  # Convert to hours then to ms
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    status = HealthStatus.WARNING
                    error_message = "Process access denied"
                    response_time_ms = 0.0
                
            else:
                # Process not found
                status = HealthStatus.DOWN
                error_message = "Process not running"
                response_time_ms = 0.0
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                error_message=error_message
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    def _check_database_health(self):
        """Check health of configured databases"""
        for db_config in self.databases_to_monitor:
            db_name = db_config['name']
            
            try:
                if db_config['type'] == 'sqlite':
                    health = self._check_sqlite_health(db_config)
                else:
                    # Placeholder for other database types
                    health = DatabaseHealth(
                        database_name=db_name,
                        status=HealthStatus.WARNING,
                        connection_time_ms=0.0,
                        query_time_ms=0.0,
                        last_check=datetime.now(),
                        error_message="Database type not supported"
                    )
                
                self.database_health[db_name] = health
                
            except Exception as e:
                logger.error(f"Error checking database {db_name}: {str(e)}")
                self.database_health[db_name] = DatabaseHealth(
                    database_name=db_name,
                    status=HealthStatus.CRITICAL,
                    connection_time_ms=0.0,
                    query_time_ms=0.0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
    
    def _check_sqlite_health(self, db_config: Dict[str, Any]) -> DatabaseHealth:
        """Check SQLite database health"""
        db_name = db_config['name']
        db_path = db_config['path']
        
        # Check if database file exists
        if not os.path.exists(db_path):
            return DatabaseHealth(
                database_name=db_name,
                status=HealthStatus.DOWN,
                connection_time_ms=0.0,
                query_time_ms=0.0,
                last_check=datetime.now(),
                error_message="Database file not found"
            )
        
        try:
            # Test connection
            connection_start = time.time()
            conn = sqlite3.connect(db_path, timeout=5.0)
            connection_time_ms = (time.time() - connection_start) * 1000
            
            # Test query
            query_start = time.time()
            cursor = conn.cursor()
            
            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Get total record count (sample from main tables)
            record_count = 0
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables[:5]:  # Limit to first 5 tables
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    record_count += cursor.fetchone()[0]
                except:
                    pass  # Skip tables with issues
            
            query_time_ms = (time.time() - query_start) * 1000
            
            conn.close()
            
            # Determine status based on performance
            if connection_time_ms > 1000 or query_time_ms > 1000:
                status = HealthStatus.WARNING
                error_message = "Slow database performance"
            else:
                status = HealthStatus.HEALTHY
                error_message = ""
            
            return DatabaseHealth(
                database_name=db_name,
                status=status,
                connection_time_ms=connection_time_ms,
                query_time_ms=query_time_ms,
                last_check=datetime.now(),
                error_message=error_message,
                table_count=table_count,
                record_count=record_count
            )
            
        except sqlite3.OperationalError as e:
            return DatabaseHealth(
                database_name=db_name,
                status=HealthStatus.CRITICAL,
                connection_time_ms=0.0,
                query_time_ms=0.0,
                last_check=datetime.now(),
                error_message=f"Database error: {str(e)}"
            )
        except Exception as e:
            return DatabaseHealth(
                database_name=db_name,
                status=HealthStatus.CRITICAL,
                connection_time_ms=0.0,
                query_time_ms=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    def _cleanup_old_metrics(self):
        """Remove old metrics to prevent memory issues"""
        if len(self.metrics_history) > self.max_history_records:
            # Keep only the most recent records
            self.metrics_history = self.metrics_history[-self.max_history_records:]
    
    def _check_alert_conditions(self, metrics: SystemMetrics):
        """Check if any metrics exceed alert thresholds"""
        alerts = []
        
        # Check CPU
        cpu_threshold = self.alert_thresholds.get("cpu_percent")
        if cpu_threshold and cpu_threshold.enabled:
            if metrics.cpu_percent > cpu_threshold.critical_threshold:
                alerts.append(f"CRITICAL: CPU usage {metrics.cpu_percent:.1f}% > {cpu_threshold.critical_threshold}%")
            elif metrics.cpu_percent > cpu_threshold.warning_threshold:
                alerts.append(f"WARNING: CPU usage {metrics.cpu_percent:.1f}% > {cpu_threshold.warning_threshold}%")
        
        # Check Memory
        memory_threshold = self.alert_thresholds.get("memory_percent")
        if memory_threshold and memory_threshold.enabled:
            if metrics.memory_percent > memory_threshold.critical_threshold:
                alerts.append(f"CRITICAL: Memory usage {metrics.memory_percent:.1f}% > {memory_threshold.critical_threshold}%")
            elif metrics.memory_percent > memory_threshold.warning_threshold:
                alerts.append(f"WARNING: Memory usage {metrics.memory_percent:.1f}% > {memory_threshold.warning_threshold}%")
        
        # Check Disk
        disk_threshold = self.alert_thresholds.get("disk_percent")
        if disk_threshold and disk_threshold.enabled:
            if metrics.disk_percent > disk_threshold.critical_threshold:
                alerts.append(f"CRITICAL: Disk usage {metrics.disk_percent:.1f}% > {disk_threshold.critical_threshold}%")
            elif metrics.disk_percent > disk_threshold.warning_threshold:
                alerts.append(f"WARNING: Disk usage {metrics.disk_percent:.1f}% > {disk_threshold.warning_threshold}%")
        
        # Check Load Average
        load_threshold = self.alert_thresholds.get("load_average")
        if load_threshold and load_threshold.enabled:
            if metrics.load_average > load_threshold.critical_threshold:
                alerts.append(f"CRITICAL: Load average {metrics.load_average:.2f} > {load_threshold.critical_threshold}")
            elif metrics.load_average > load_threshold.warning_threshold:
                alerts.append(f"WARNING: Load average {metrics.load_average:.2f} > {load_threshold.warning_threshold}")
        
        # Log alerts
        for alert in alerts:
            if "CRITICAL" in alert:
                logger.critical(alert)
            else:
                logger.warning(alert)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status summary"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Overall system health
        overall_status = HealthStatus.HEALTHY
        
        # Check service health
        service_statuses = list(self.service_health.values())
        if any(s.status == HealthStatus.DOWN for s in service_statuses):
            overall_status = HealthStatus.DOWN
        elif any(s.status == HealthStatus.CRITICAL for s in service_statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(s.status == HealthStatus.WARNING for s in service_statuses):
            overall_status = HealthStatus.WARNING
        
        # Check database health
        db_statuses = list(self.database_health.values())
        if any(d.status in [HealthStatus.DOWN, HealthStatus.CRITICAL] for d in db_statuses):
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.CRITICAL
        
        status = {
            'overall_status': overall_status.value,
            'monitoring_active': self.is_monitoring,
            'last_update': datetime.now().isoformat(),
            'system_metrics': asdict(latest_metrics) if latest_metrics else None,
            'services': {name: asdict(health) for name, health in self.service_health.items()},
            'databases': {name: asdict(health) for name, health in self.database_health.items()},
            'uptime_hours': latest_metrics.uptime_seconds / 3600 if latest_metrics else 0,
            'metrics_history_count': len(self.metrics_history)
        }
        
        return status
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            asdict(m) for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        return recent_metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        current_status = self.get_current_status()
        
        # Calculate service availability
        service_availability = {}
        for name, health in self.service_health.items():
            if health.status == HealthStatus.HEALTHY:
                availability = 100.0
            elif health.status == HealthStatus.WARNING:
                availability = 75.0
            elif health.status == HealthStatus.CRITICAL:
                availability = 25.0
            else:
                availability = 0.0
            service_availability[name] = availability
        
        # Calculate database performance
        db_performance = {}
        for name, health in self.database_health.items():
            avg_query_time = health.query_time_ms
            if avg_query_time < 100:
                performance = "Excellent"
            elif avg_query_time < 500:
                performance = "Good"
            elif avg_query_time < 1000:
                performance = "Fair"
            else:
                performance = "Poor"
            db_performance[name] = {
                'performance': performance,
                'query_time_ms': avg_query_time,
                'connection_time_ms': health.connection_time_ms
            }
        
        summary = {
            'system_status': current_status,
            'service_availability': service_availability,
            'database_performance': db_performance,
            'resource_utilization': {
                'cpu_percent': current_status['system_metrics']['cpu_percent'] if current_status['system_metrics'] else 0,
                'memory_percent': current_status['system_metrics']['memory_percent'] if current_status['system_metrics'] else 0,
                'disk_percent': current_status['system_metrics']['disk_percent'] if current_status['system_metrics'] else 0
            },
            'alert_summary': {
                'total_thresholds': len(self.alert_thresholds),
                'enabled_thresholds': len([t for t in self.alert_thresholds.values() if t.enabled])
            }
        }
        
        return summary
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics history to file"""
        metrics_data = self.get_metrics_history(hours)
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump({
                    'export_time': datetime.now().isoformat(),
                    'hours_exported': hours,
                    'metrics_count': len(metrics_data),
                    'metrics': metrics_data,
                    'services': {name: asdict(health) for name, health in self.service_health.items()},
                    'databases': {name: asdict(health) for name, health in self.database_health.items()}
                }, f, indent=2, default=str)
            
            logger.info(f"Exported {len(metrics_data)} metrics to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'check_interval': 10,  # Check every 10 seconds for testing
        'services': [
            {'name': 'trading_api', 'url': 'http://localhost:8000/health', 'timeout': 5},
            {'name': 'python_process', 'check_type': 'process', 'process_name': 'python'}
        ],
        'databases': [
            {'name': 'trading_db', 'path': 'trading_data.db', 'type': 'sqlite'}
        ]
    }
    
    # Create and start monitor
    monitor = SystemMonitor(config)
    monitor.start_monitoring()
    
    try:
        # Run for a short time for testing
        time.sleep(30)
        
        # Get status
        status = monitor.get_current_status()
        print("System Status:", status['overall_status'])
        
        # Get health summary
        summary = monitor.get_health_summary()
        print("Health Summary:", summary)
        
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop_monitoring()