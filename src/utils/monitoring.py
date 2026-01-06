"""Monitoring and observability utilities."""

import time
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class Metric:
    """A monitored metric."""

    name: str
    value: float
    timestamp: float
    tags: dict[str, str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = {}


class MetricBuffer:
    """Buffer for storing metrics."""

    def __init__(self, max_size: int = 10000):
        """
        Initialize metric buffer.

        Args:
            max_size: Maximum number of metrics to store.
        """
        self.max_size = max_size
        self.metrics: list[Metric] = []
        self._lock = None

    def add(self, metric: Metric) -> None:
        """
        Add metric to buffer.

        Args:
            metric: Metric to add.
        """
        # Thread-safe if lock is available
        if self._lock:
            with self._lock:
                self._add_metric(metric)
        else:
            self._add_metric(metric)

    def _add_metric(self, metric: Metric) -> None:
        """Internal method to add metric."""
        self.metrics.append(metric)

        # Remove old metrics if buffer is full
        if len(self.metrics) > self.max_size:
            self.metrics = self.metrics[-self.max_size :]

    def get_recent(self, n: int = 100) -> list[Metric]:
        """
        Get recent metrics.

        Args:
            n: Number of recent metrics to get.

        Returns:
            List of recent metrics.
        """
        if self._lock:
            with self._lock:
                return self.metrics[-n:]
        else:
            return self.metrics[-n:]

    def clear(self) -> None:
        """Clear all metrics."""
        if self._lock:
            with self._lock:
                self.metrics.clear()
        else:
            self.metrics.clear()


class PlatformMonitor:
    """Platform monitoring with alerting."""

    def __init__(self, alert_thresholds: dict[str, float] = None):
        """
        Initialize platform monitor.

        Args:
            alert_thresholds: Thresholds for alerts.
        """
        self.metrics = MetricBuffer()
        self.alerts: list[dict[str, Any]] = []
        self.alert_thresholds = alert_thresholds or {
            "high_memory_usage": 0.9,  # 90% memory usage
            "high_cpu_usage": 0.8,  # 80% CPU usage
            "training_loss_spike": 5.0,  # 5x increase in loss
            "prediction_latency": 1.0,  # 1 second latency
        }
        self._running = False
        self._monitor_thread: Thread | None = None

    def start(self, interval: float = 60.0) -> None:
        """
        Start monitoring thread.

        Args:
            interval: Monitoring interval in seconds.
        """
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._monitor_thread = Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"Started monitoring with {interval}s interval")

    def stop(self) -> None:
        """Stop monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Stopped monitoring")

    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self.collect_system_metrics()
                self.check_alerts()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            time.sleep(interval)

    def collect_system_metrics(self) -> None:
        """Collect system metrics."""
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.record_metric("system.cpu.percent", cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric("system.memory.percent", memory.percent)
        self.record_metric("system.memory.used_gb", memory.used / 1e9)
        self.record_metric("system.memory.available_gb", memory.available / 1e9)

        # Disk usage
        disk = psutil.disk_usage("/")
        self.record_metric("system.disk.percent", disk.percent)

        # Network I/O
        net_io = psutil.net_io_counters()
        self.record_metric("system.network.bytes_sent", net_io.bytes_sent)
        self.record_metric("system.network.bytes_recv", net_io.bytes_recv)

        # Process info
        process = psutil.Process()
        self.record_metric("process.memory.rss_gb", process.memory_info().rss / 1e9)
        self.record_metric("process.cpu.percent", process.cpu_percent())

    def record_metric(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """
        Record a metric.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Metric tags.
        """
        metric = Metric(name=name, value=value, timestamp=time.time(), tags=tags or {})

        self.metrics.add(metric)

    def log_training_metrics(
        self, metrics: dict[str, float], context: str = ""
    ) -> None:
        """
        Log training metrics with anomaly detection.

        Args:
            metrics: Training metrics.
            context: Training context.
        """
        for name, value in metrics.items():
            self.record_metric(f"training.{name}", value, tags={"context": context})

        # Check for anomalies
        self._check_training_anomalies(metrics, context)

    def _check_training_anomalies(
        self, metrics: dict[str, float], context: str
    ) -> None:
        """Check for training anomalies."""
        # Check for loss spikes
        if "loss" in metrics:
            recent_losses = [
                m.value
                for m in self.metrics.get_recent(10)
                if m.name == "training.loss" and m.tags.get("context") == context
            ]

            if len(recent_losses) >= 2:
                current_loss = metrics["loss"]
                avg_previous_loss = np.mean(recent_losses[:-1])

                if avg_previous_loss > 0 and current_loss > avg_previous_loss * 5:
                    self.raise_alert(
                        "training_loss_spike",
                        f"Training loss spike in {context}: "
                        f"{current_loss:.4f} (5x previous average)",
                        severity="warning",
                    )

    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """
        Detect data drift between reference and current data.

        Args:
            reference_data: Reference data.
            current_data: Current data.
            feature_names: Feature names.

        Returns:
            Drift scores for each feature.
        """
        from scipy import stats

        drift_scores = {}

        for i, feature_name in enumerate(feature_names):
            try:
                # Kolmogorov-Smirnov test for distribution drift
                ks_stat, ks_pvalue = stats.ks_2samp(
                    reference_data[:, i], current_data[:, i]
                )

                drift_scores[feature_name] = {
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue),
                    "drift_detected": ks_pvalue < 0.05,  # 95% confidence
                }

                # Record drift metric
                self.record_metric(f"drift.{feature_name}.ks_statistic", float(ks_stat))

                if ks_pvalue < 0.05:
                    self.raise_alert(
                        "data_drift",
                        f"Data drift detected in feature {feature_name}: "
                        f"KS statistic = {ks_stat:.4f}, p-value = {ks_pvalue:.4f}",
                        severity="warning",
                        tags={"feature": feature_name},
                    )

            except Exception as e:
                logger.warning(f"Could not compute drift for {feature_name}: {e}")

        return drift_scores

    def model_performance_decay(
        self,
        current_metrics: dict[str, float],
        reference_metrics: dict[str, float],
        model_version: str,
    ) -> dict[str, float]:
        """
        Monitor model performance decay.

        Args:
            current_metrics: Current model metrics.
            reference_metrics: Reference model metrics.
            model_version: Model version.

        Returns:
            Performance decay scores.
        """
        decay_scores = {}

        for metric_name in set(current_metrics.keys()) & set(reference_metrics.keys()):
            current_value = current_metrics[metric_name]
            reference_value = reference_metrics[metric_name]

            if reference_value != 0:
                decay_ratio = abs(current_value - reference_value) / abs(
                    reference_value
                )
                decay_scores[metric_name] = decay_ratio

                self.record_metric(
                    f"model.{model_version}.decay.{metric_name}", decay_ratio
                )

                if decay_ratio > 0.1:  # 10% decay
                    self.raise_alert(
                        "model_decay",
                        f"Model {model_version} performance decay in {metric_name}: "
                        f"{decay_ratio:.1%} decay",
                        severity="warning",
                        tags={"model_version": model_version, "metric": metric_name},
                    )

        return decay_scores

    def raise_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Raise an alert.

        Args:
            alert_type: Type of alert.
            message: Alert message.
            severity: Alert severity.
            tags: Alert tags.
        """
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {},
        }

        self.alerts.append(alert)

        # Log based on severity
        if severity == "critical":
            logger.critical(f"ALERT: {message}", extra=alert)
        elif severity == "error":
            logger.error(f"ALERT: {message}", extra=alert)
        elif severity == "warning":
            logger.warning(f"ALERT: {message}", extra=alert)
        else:
            logger.info(f"ALERT: {message}", extra=alert)

    def check_alerts(self) -> None:
        """Check for threshold-based alerts."""
        recent_metrics = self.metrics.get_recent(100)

        for metric in recent_metrics:
            # Check CPU usage
            if metric.name == "system.cpu.percent":
                if metric.value > self.alert_thresholds["high_cpu_usage"] * 100:
                    self.raise_alert(
                        "high_cpu_usage",
                        f"High CPU usage: {metric.value:.1f}%",
                        severity="warning",
                    )

            # Check memory usage
            elif metric.name == "system.memory.percent":
                if metric.value > self.alert_thresholds["high_memory_usage"] * 100:
                    self.raise_alert(
                        "high_memory_usage",
                        f"High memory usage: {metric.value:.1f}%",
                        severity="warning",
                    )

            # Check prediction latency
            elif metric.name == "serving.prediction.latency":
                if metric.value > self.alert_thresholds["prediction_latency"]:
                    self.raise_alert(
                        "high_prediction_latency",
                        f"High prediction latency: {metric.value:.3f}s",
                        severity="warning",
                    )

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary."""
        recent_metrics = self.metrics.get_recent(100)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.metrics.metrics),
            "recent_metrics_count": len(recent_metrics),
            "active_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else [],
            "system_metrics": {},
        }

        # Aggregate recent metrics
        for metric in recent_metrics:
            if metric.name.startswith("system."):
                if metric.name not in summary["system_metrics"]:
                    summary["system_metrics"][metric.name] = []
                summary["system_metrics"][metric.name].append(metric.value)

        # Compute averages
        for name, values in summary["system_metrics"].items():
            summary["system_metrics"][f"{name}.avg"] = float(np.mean(values))
            summary["system_metrics"][f"{name}.max"] = float(np.max(values))

        return summary


# Global monitor instance
_monitor_instance: PlatformMonitor | None = None


def get_monitor() -> PlatformMonitor:
    """
    Get global monitor instance.

    Returns:
        PlatformMonitor instance.
    """
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = PlatformMonitor()

    return _monitor_instance


def start_monitoring(interval: float = 60.0) -> None:
    """
    Start global monitoring.

    Args:
        interval: Monitoring interval.
    """
    monitor = get_monitor()
    monitor.start(interval)


def stop_monitoring() -> None:
    """Stop global monitoring."""
    monitor = get_monitor()
    monitor.stop()
