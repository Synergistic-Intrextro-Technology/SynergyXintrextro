/**
 * Adaptive Strategy Interface
 * Embodies the principle: "Use multiple strategies, and performance-based learning"
 */
export interface Strategy {
  readonly name: string;
  execute<T>(context: T): Promise<T>;
  getMetrics(): StrategyMetrics;
}

export interface StrategyMetrics {
  executionCount: number;
  averageExecutionTime: number;
  successRate: number;
  lastExecuted: Date | null;
}

/**
 * Adaptive System Core
 * Demonstrates: "Code is capacity: to act, adapt, and evolve"
 */
export class AdaptiveSystem {
  private strategies: Map<string, Strategy> = new Map();
  private performanceHistory: Map<string, number[]> = new Map();

  /**
   * Register a strategy for adaptive execution
   * Supports: "systems live, experiment, and balance stability with novelty"
   */
  registerStrategy(strategy: Strategy): void {
    this.strategies.set(strategy.name, strategy);
    this.performanceHistory.set(strategy.name, []);
  }

  /**
   * Execute with adaptive strategy selection
   * Implements: "performance-based learning"
   */
  async execute<T>(context: T): Promise<T> {
    const bestStrategy = this.selectBestStrategy();
    
    if (!bestStrategy) {
      throw new Error('No strategies available for execution');
    }

    const startTime = Date.now();
    const result = await bestStrategy.execute(context);
    const executionTime = Date.now() - startTime;

    // Record performance for learning
    this.recordPerformance(bestStrategy.name, executionTime);

    return result;
  }

  /**
   * Select strategy based on performance metrics
   * Demonstrates: "Let live metrics guide adaptation"
   */
  private selectBestStrategy(): Strategy | null {
    if (this.strategies.size === 0) return null;

    let bestStrategy: Strategy | null = null;
    let bestScore = -1;

    for (const [, strategy] of this.strategies) {
      const metrics = strategy.getMetrics();
      const score = this.calculateStrategyScore(metrics);
      
      if (score > bestScore) {
        bestScore = score;
        bestStrategy = strategy;
      }
    }

    return bestStrategy;
  }

  /**
   * Calculate strategy performance score
   * Balances: "stability with novelty"
   */
  private calculateStrategyScore(metrics: StrategyMetrics): number {
    const stabilityScore = metrics.successRate * 0.4;
    const performanceScore = Math.max(0, (1000 - metrics.averageExecutionTime) / 1000) * 0.4;
    const noveltyScore = metrics.executionCount === 0 ? 0.2 : 0; // Give new strategies a chance
    
    return stabilityScore + performanceScore + noveltyScore;
  }

  private recordPerformance(strategyName: string, executionTime: number): void {
    const history = this.performanceHistory.get(strategyName) || [];
    history.push(executionTime);
    
    // Keep only recent history for adaptation
    if (history.length > 100) {
      history.shift();
    }
    
    this.performanceHistory.set(strategyName, history);
  }

  /**
   * Get system health metrics
   * Supports: "transparency" and "self-reflection"
   */
  getSystemMetrics(): SystemMetrics {
    const strategies = Array.from(this.strategies.values()).map(s => s.getMetrics());
    const totalExecutions = strategies.reduce((sum, s) => sum + s.executionCount, 0);
    const averageSuccessRate = strategies.length > 0 
      ? strategies.reduce((sum, s) => sum + s.successRate, 0) / strategies.length 
      : 0;

    return {
      totalStrategies: strategies.length,
      totalExecutions,
      systemSuccessRate: averageSuccessRate,
      lastAdaptation: new Date()
    };
  }
}

export interface SystemMetrics {
  totalStrategies: number;
  totalExecutions: number;
  systemSuccessRate: number;
  lastAdaptation: Date;
}