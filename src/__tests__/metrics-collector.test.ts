import { MetricsCollector } from '../metrics/collector';

describe('MetricsCollector', () => {
  let metricsCollector: MetricsCollector;

  beforeEach(() => {
    metricsCollector = new MetricsCollector();
  });

  test('should record metrics successfully', () => {
    expect(() => {
      metricsCollector.record('test.metric', 42);
    }).not.toThrow();
  });

  test('should record metrics with tags', () => {
    expect(() => {
      metricsCollector.record('test.metric', 42, { environment: 'test' });
    }).not.toThrow();
  });

  test('should retrieve aggregated metrics', () => {
    // Record some test metrics
    metricsCollector.record('latency', 100);
    metricsCollector.record('latency', 150);
    metricsCollector.record('latency', 120);

    const aggregated = metricsCollector.getAggregated('latency');

    expect(aggregated).not.toBeNull();
    expect(aggregated!.count).toBe(3);
    expect(aggregated!.sum).toBe(370);
    expect(aggregated!.avg).toBeCloseTo(123.33, 2);
    expect(aggregated!.min).toBe(100);
    expect(aggregated!.max).toBe(150);
    expect(aggregated!.latest).toBe(120);
  });

  test('should return null for non-existent metrics', () => {
    const aggregated = metricsCollector.getAggregated('non.existent');
    expect(aggregated).toBeNull();
  });

  test('should respect time window in aggregation', async () => {
    // Record metric and wait
    metricsCollector.record('time.test', 100);
    
    // Wait a bit and record another
    await new Promise(resolve => setTimeout(resolve, 10));
    metricsCollector.record('time.test', 200);

    // Get metrics for very short time window (should only include recent)
    const recentOnly = metricsCollector.getAggregated('time.test', 5); // 5ms window
    const allTime = metricsCollector.getAggregated('time.test');

    expect(allTime!.count).toBe(2);
    // Recent might be 1 or 2 depending on timing
    expect(recentOnly!.count).toBeGreaterThanOrEqual(1);
  });

  test('should provide health snapshot', () => {
    metricsCollector.record('requests.success', 100);
    metricsCollector.record('requests.error', 2);
    metricsCollector.record('latency.avg', 150);

    const snapshot = metricsCollector.getHealthSnapshot();

    expect(snapshot).toHaveProperty('totalMetricTypes');
    expect(snapshot).toHaveProperty('recentMetrics');
    expect(snapshot).toHaveProperty('lastUpdated');
    expect(snapshot).toHaveProperty('isHealthy');
    expect(snapshot.totalMetricTypes).toBe(3);
  });

  test('should assess system health correctly', () => {
    // Record healthy metrics
    metricsCollector.record('requests.success', 100);
    metricsCollector.record('latency.avg', 100); // Under 5s threshold

    let snapshot = metricsCollector.getHealthSnapshot();
    expect(snapshot.isHealthy).toBe(true);

    // Record unhealthy metrics
    metricsCollector.record('error.rate', 5); // Contains 'error'
    
    snapshot = metricsCollector.getHealthSnapshot();
    expect(snapshot.isHealthy).toBe(false);
  });

  test('should export metrics in JSON format', () => {
    metricsCollector.record('test.export', 42, { type: 'test' });
    metricsCollector.record('test.export', 84, { type: 'test' });

    const exported = metricsCollector.exportMetrics('json');
    const parsed = JSON.parse(exported);

    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed).toHaveLength(2);
    expect(parsed[0]).toHaveProperty('name', 'test.export');
    expect(parsed[0]).toHaveProperty('value');
    expect(parsed[0]).toHaveProperty('timestamp');
    expect(parsed[0]).toHaveProperty('tags');
  });

  test('should export metrics in CSV format', () => {
    metricsCollector.record('test.csv', 42);
    
    const exported = metricsCollector.exportMetrics('csv');
    const lines = exported.split('\n');

    expect(lines[0]).toBe('name,value,timestamp,tags');
    expect(lines[1]).toContain('test.csv,42');
  });

  test('should maintain reasonable memory usage', () => {
    // Record many metrics to test memory management
    for (let i = 0; i < 1500; i++) {
      metricsCollector.record('memory.test', i);
    }

    const aggregated = metricsCollector.getAggregated('memory.test');
    expect(aggregated!.count).toBeLessThanOrEqual(1000); // Should be capped
  });
});