import { AdaptiveSystem, Strategy, StrategyMetrics } from '../core/adaptive-system';

// Mock strategy for testing
class MockStrategy implements Strategy {
  public readonly name: string;
  private executionCount = 0;
  private successCount = 0;
  private avgTime: number;

  constructor(name: string, avgTime: number = 50) {
    this.name = name;
    this.avgTime = avgTime;
  }

  async execute<T>(context: T): Promise<T> {
    this.executionCount++;
    
    // Simulate variable execution time
    await new Promise(resolve => 
      setTimeout(resolve, this.avgTime + Math.random() * 20 - 10)
    );
    
    // Simulate occasional failures for realism
    if (Math.random() > 0.1) { // 90% success rate
      this.successCount++;
    }
    
    return context;
  }

  getMetrics(): StrategyMetrics {
    return {
      executionCount: this.executionCount,
      averageExecutionTime: this.avgTime,
      successRate: this.executionCount > 0 ? this.successCount / this.executionCount : 0,
      lastExecuted: new Date()
    };
  }
}

describe('AdaptiveSystem', () => {
  let adaptiveSystem: AdaptiveSystem;

  beforeEach(() => {
    adaptiveSystem = new AdaptiveSystem();
  });

  test('should register strategies successfully', () => {
    const strategy = new MockStrategy('test-strategy');
    
    expect(() => adaptiveSystem.registerStrategy(strategy)).not.toThrow();
  });

  test('should execute with registered strategy', async () => {
    const strategy = new MockStrategy('fast-strategy', 10);
    adaptiveSystem.registerStrategy(strategy);
    
    const testData = { value: 'test' };
    const result = await adaptiveSystem.execute(testData);
    
    expect(result).toEqual(testData);
  });

  test('should throw error when no strategies are available', async () => {
    const testData = { value: 'test' };
    
    await expect(adaptiveSystem.execute(testData)).rejects.toThrow('No strategies available for execution');
  });

  test('should select best performing strategy', async () => {
    const fastStrategy = new MockStrategy('fast', 10);
    const slowStrategy = new MockStrategy('slow', 100);
    
    adaptiveSystem.registerStrategy(fastStrategy);
    adaptiveSystem.registerStrategy(slowStrategy);
    
    // Execute multiple times to establish performance patterns
    for (let i = 0; i < 5; i++) {
      await adaptiveSystem.execute({ iteration: i });
    }
    
    const metrics = adaptiveSystem.getSystemMetrics();
    expect(metrics.totalStrategies).toBe(2);
    expect(metrics.totalExecutions).toBeGreaterThan(0);
  });

  test('should provide system metrics', () => {
    const strategy = new MockStrategy('test');
    adaptiveSystem.registerStrategy(strategy);
    
    const metrics = adaptiveSystem.getSystemMetrics();
    
    expect(metrics).toHaveProperty('totalStrategies');
    expect(metrics).toHaveProperty('totalExecutions');
    expect(metrics).toHaveProperty('systemSuccessRate');
    expect(metrics).toHaveProperty('lastAdaptation');
    expect(metrics.totalStrategies).toBe(1);
  });

  test('should demonstrate adaptive behavior with multiple strategies', async () => {
    // Create strategies with different performance characteristics
    const strategies = [
      new MockStrategy('strategy-1', 30),
      new MockStrategy('strategy-2', 20),
      new MockStrategy('strategy-3', 40)
    ];
    
    strategies.forEach(s => adaptiveSystem.registerStrategy(s));
    
    // Execute multiple times to allow adaptation
    const results = [];
    for (let i = 0; i < 10; i++) {
      const result = await adaptiveSystem.execute({ test: i });
      results.push(result);
    }
    
    expect(results).toHaveLength(10);
    
    const finalMetrics = adaptiveSystem.getSystemMetrics();
    expect(finalMetrics.totalExecutions).toBe(10);
    expect(finalMetrics.totalStrategies).toBe(3);
  });
});