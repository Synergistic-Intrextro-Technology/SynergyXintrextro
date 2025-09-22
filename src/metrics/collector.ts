/**
 * Metrics Collection System
 * Implements: "Let live metrics guide adaptation"
 */
export interface Metric {
  name: string;
  value: number;
  timestamp: Date;
  tags?: Record<string, string>;
}

export interface MetricAggregation {
  count: number;
  sum: number;
  avg: number;
  min: number;
  max: number;
  latest: number;
}

/**
 * Live Metrics Collector
 * Supports: "performance-based learning" and "transparency"
 */
export class MetricsCollector {
  private metrics: Map<string, Metric[]> = new Map();
  private readonly maxMetricsPerKey = 1000; // Prevent memory bloat

  /**
   * Record a metric for system learning
   */
  record(name: string, value: number, tags?: Record<string, string>): void {
    const metric: Metric = {
      name,
      value,
      timestamp: new Date(),
      tags
    };

    const existing = this.metrics.get(name) || [];
    existing.push(metric);

    // Maintain reasonable history size
    if (existing.length > this.maxMetricsPerKey) {
      existing.shift(); // Remove oldest
    }

    this.metrics.set(name, existing);
  }

  /**
   * Get aggregated metrics for decision making
   * Enables: "Let live metrics guide adaptation"
   */
  getAggregated(name: string, timeWindowMs?: number): MetricAggregation | null {
    const metrics = this.metrics.get(name);
    if (!metrics || metrics.length === 0) return null;

    const cutoff = timeWindowMs ? new Date(Date.now() - timeWindowMs) : null;
    const relevantMetrics = cutoff 
      ? metrics.filter(m => m.timestamp >= cutoff)
      : metrics;

    if (relevantMetrics.length === 0) return null;

    const values = relevantMetrics.map(m => m.value);
    
    return {
      count: values.length,
      sum: values.reduce((a, b) => a + b, 0),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      latest: values[values.length - 1]
    };
  }

  /**
   * Get current system health snapshot
   * Promotes: "transparency" and "self-reflection"
   */
  getHealthSnapshot(): HealthSnapshot {
    const metricNames = Array.from(this.metrics.keys());
    const recentMetrics = metricNames.map(name => {
      const agg = this.getAggregated(name, 60000); // Last minute
      return { name, aggregation: agg };
    }).filter(m => m.aggregation !== null);

    return {
      totalMetricTypes: metricNames.length,
      recentMetrics: recentMetrics as HealthMetric[],
      lastUpdated: new Date(),
      isHealthy: this.assessSystemHealth(recentMetrics as HealthMetric[])
    };
  }

  /**
   * Assess system health based on metrics
   * Demonstrates: "Engineer for clarity, security, and maintainability"
   */
  private assessSystemHealth(metrics: HealthMetric[]): boolean {
    // Simple health assessment - can be made more sophisticated
    for (const metric of metrics) {
      if (metric.name.includes('error') && (metric.aggregation?.latest ?? 0) > 0) {
        return false;
      }
      if (metric.name.includes('latency') && (metric.aggregation?.avg ?? 0) > 5000) {
        return false; // More than 5s average latency
      }
    }
    return true;
  }

  /**
   * Clear old metrics for memory management
   * Supports: "maintainability"
   */
  cleanup(olderThanMs: number = 24 * 60 * 60 * 1000): void { // Default 24 hours
    const cutoff = new Date(Date.now() - olderThanMs);
    
    for (const [name, metrics] of this.metrics) {
      const filtered = metrics.filter(m => m.timestamp >= cutoff);
      this.metrics.set(name, filtered);
    }
  }

  /**
   * Export metrics for external analysis
   * Enables: "collaboration" and "transparency"
   */
  exportMetrics(format: 'json' | 'csv' = 'json'): string {
    const allMetrics = Array.from(this.metrics.values()).flat();
    
    if (format === 'csv') {
      const headers = 'name,value,timestamp,tags\n';
      const rows = allMetrics.map(m => 
        `${m.name},${m.value},${m.timestamp.toISOString()},${JSON.stringify(m.tags || {})}`
      ).join('\n');
      return headers + rows;
    }
    
    return JSON.stringify(allMetrics, null, 2);
  }
}

interface HealthMetric {
  name: string;
  aggregation: MetricAggregation;
}

export interface HealthSnapshot {
  totalMetricTypes: number;
  recentMetrics: HealthMetric[];
  lastUpdated: Date;
  isHealthy: boolean;
}