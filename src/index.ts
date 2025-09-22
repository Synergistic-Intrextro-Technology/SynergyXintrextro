/**
 * SynergyXintrextro Framework
 * Embodies: "Code is capacity: to act, adapt, and evolve"
 */
import { AdaptiveSystem, Strategy, StrategyMetrics } from './core/adaptive-system';
import { MetricsCollector } from './metrics/collector';
import { QualityController, QualityCheck, QualityResult, QualityGate } from './quality/controller';
import { CollaborationHub } from './collaboration/hub';

/**
 * Example Strategy Implementation
 * Demonstrates: "Use multiple strategies, and performance-based learning"
 */
class OptimizedStrategy implements Strategy {
  public readonly name = 'optimized';
  private executionCount = 0;
  private totalTime = 0;
  private successCount = 0;
  private lastExecuted: Date | null = null;

  async execute<T>(context: T): Promise<T> {
    const start = Date.now();
    this.executionCount++;
    this.lastExecuted = new Date();

    try {
      // Simulate optimized processing
      await this.processOptimized();
      this.successCount++;
      return context;
    } finally {
      this.totalTime += Date.now() - start;
    }
  }

  private async processOptimized(): Promise<void> {
    // Simulate fast, optimized processing
    await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 20));
  }

  getMetrics(): StrategyMetrics {
    return {
      executionCount: this.executionCount,
      averageExecutionTime: this.executionCount > 0 ? this.totalTime / this.executionCount : 0,
      successRate: this.executionCount > 0 ? this.successCount / this.executionCount : 0,
      lastExecuted: this.lastExecuted
    };
  }
}

class StableStrategy implements Strategy {
  public readonly name = 'stable';
  private executionCount = 0;
  private totalTime = 0;
  private successCount = 0;
  private lastExecuted: Date | null = null;

  async execute<T>(context: T): Promise<T> {
    const start = Date.now();
    this.executionCount++;
    this.lastExecuted = new Date();

    try {
      // Simulate stable but slower processing
      await this.processStable();
      this.successCount++;
      return context;
    } finally {
      this.totalTime += Date.now() - start;
    }
  }

  private async processStable(): Promise<void> {
    // Simulate stable, reliable processing
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 30));
  }

  getMetrics(): StrategyMetrics {
    return {
      executionCount: this.executionCount,
      averageExecutionTime: this.executionCount > 0 ? this.totalTime / this.executionCount : 0,
      successRate: this.executionCount > 0 ? this.successCount / this.executionCount : 0,
      lastExecuted: this.lastExecuted
    };
  }
}

/**
 * Example Quality Check
 * Implements: "Engineer for clarity, security, and maintainability"
 */
class SecurityCheck implements QualityCheck {
  public readonly name = 'security';
  public readonly description = 'Validates security best practices';

  async execute<T>(input: T): Promise<QualityResult> {
    // Simulate security validation
    const inputStr = JSON.stringify(input);
    const issues: string[] = [];
    let score = 100;

    // Check for common security issues
    if (inputStr.includes('password')) {
      issues.push('Potential password exposure detected');
      score -= 30;
    }
    // Improved check for dangerous eval usage (e.g., eval(, eval (, window.eval(, etc.)
    const evalPattern = /\b(?:window\.)?eval\s*\(/i;
    if (evalPattern.test(inputStr)) {
      issues.push('Dangerous eval() usage detected');
      score -= 50;
    }
    if (inputStr.length > 10000) {
      issues.push('Input size may pose DoS risk');
      score -= 20;
    }

    return {
      passed: score >= 70,
      score: Math.max(0, score),
      message: issues.length > 0 ? issues.join('; ') : 'Security check passed',
      suggestions: issues.length > 0 ? ['Review security guidelines', 'Implement input validation'] : undefined,
      metrics: {
        inputSize: inputStr.length,
        issuesFound: issues.length
      }
    };
  }
}

class PerformanceCheck implements QualityCheck {
  public readonly name = 'performance';
  public readonly description = 'Validates performance characteristics';

  async execute<T>(input: T): Promise<QualityResult> {
    const start = Date.now();
    
    // Simulate performance analysis
    await new Promise(resolve => setTimeout(resolve, 5 + Math.random() * 10));
    
    const duration = Date.now() - start;
    const inputSize = JSON.stringify(input).length;
    
    let score = 100;
    const suggestions: string[] = [];

    if (duration > 100) {
      score -= 40;
      suggestions.push('Optimize processing time');
    }
    if (inputSize > 5000) {
      score -= 20;
      suggestions.push('Consider data compression');
    }

    return {
      passed: score >= 60,
      score,
      message: `Performance check completed in ${duration}ms`,
      suggestions: suggestions.length > 0 ? suggestions : undefined,
      metrics: {
        processingTime: duration,
        inputSize
      }
    };
  }
}

/**
 * Framework Integration
 * Demonstrates: "True inventiveness is disciplined evolution—purposeful, sustainable, and open to discovery"
 */
export class SynergyFramework {
  private adaptiveSystem: AdaptiveSystem;
  private metricsCollector: MetricsCollector;
  private qualityController: QualityController;
  private collaborationHub: CollaborationHub;

  constructor() {
    this.adaptiveSystem = new AdaptiveSystem();
    this.metricsCollector = new MetricsCollector();
    this.qualityController = new QualityController();
    this.collaborationHub = new CollaborationHub();

    this.initialize();
  }

  private initialize(): void {
    // Register strategies for adaptation
    this.adaptiveSystem.registerStrategy(new OptimizedStrategy());
    this.adaptiveSystem.registerStrategy(new StableStrategy());

    // Set up quality gates
    const mainGate: QualityGate = {
      name: 'main',
      checks: [new SecurityCheck(), new PerformanceCheck()],
      minimumScore: 70,
      required: true
    };
    this.qualityController.registerGate(mainGate);

    // Add foundational knowledge
    this.collaborationHub.addKnowledge({
      title: 'Framework Philosophy',
      content: 'Code is capacity: to act, adapt, and evolve. Build only what strengthens control, stability, and quality.',
      author: 'system',
      tags: ['philosophy', 'core-principles']
    });
  }

  /**
   * Process data through the adaptive system
   * Demonstrates: "systems live, experiment, and balance stability with novelty"
   */
  async process<T>(data: T): Promise<ProcessResult<T>> {
    const startTime = Date.now();

    // Quality gate first - ensure standards
    const qualityResult = await this.qualityController.executeGate('main', data);
    
    if (!qualityResult.passed) {
      this.metricsCollector.record('quality.failures', 1);
      return {
        success: false,
        data,
        qualityResult,
        message: 'Quality gate failed',
        processingTime: Date.now() - startTime
      };
    }

    // Adaptive processing
    try {
      const processedData = await this.adaptiveSystem.execute(data);
      const processingTime = Date.now() - startTime;

      // Record metrics for learning
      this.metricsCollector.record('processing.time', processingTime);
      this.metricsCollector.record('processing.success', 1);

      return {
        success: true,
        data: processedData,
        qualityResult,
        processingTime,
        systemMetrics: this.adaptiveSystem.getSystemMetrics()
      };
    } catch (error) {
      this.metricsCollector.record('processing.errors', 1);
      
      return {
        success: false,
        data,
        qualityResult,
        message: `Processing failed: ${error}`,
        processingTime: Date.now() - startTime
      };
    }
  }

  /**
   * Get comprehensive system status
   * Promotes: "transparency" and "self-reflection"
   */
  getSystemStatus(): SystemStatus {
    return {
      adaptive: this.adaptiveSystem.getSystemMetrics(),
      health: this.metricsCollector.getHealthSnapshot(),
      quality: this.qualityController.getQualityTrends(),
      collaboration: this.collaborationHub.getCollaborationInsights(),
      timestamp: new Date()
    };
  }

  /**
   * Get improvement recommendations
   * Embodies: "disciplined evolution"
   */
  getRecommendations(): Recommendation[] {
    const recommendations: Recommendation[] = [];
    
    // Quality-based recommendations
    const qualityImprovements = this.qualityController.getImprovementSuggestions();
    recommendations.push(...qualityImprovements.map(q => ({
      category: 'quality',
      priority: q.priority,
      description: q.description,
      action: q.action
    })));

    // System health recommendations
    const health = this.metricsCollector.getHealthSnapshot();
    if (!health.isHealthy) {
      recommendations.push({
        category: 'health',
        priority: 'high',
        description: 'System health indicators show issues',
        action: 'Review metrics and address underlying problems'
      });
    }

    return recommendations;
  }
}

// --- Type-safe interfaces for system status and metrics ---

// Example: Based on usage in demonstrateFramework and getSystemStatus
export interface AdaptiveStatus {
  totalStrategies: number;
  [key: string]: any; // Extend as needed
}

export interface HealthStatus {
  isHealthy: boolean;
  [key: string]: any; // Extend as needed
}

export interface QualityStatus {
  trend: string;
  [key: string]: any; // Extend as needed
}

export interface CollaborationStatus {
  [key: string]: any; // Define properties as needed
}

export interface ProcessResult<T> {
  success: boolean;
  data: T;
  qualityResult: QualityResult;
  message?: string;
  processingTime: number;
  systemMetrics?: StrategyMetrics;
}

export interface SystemStatus {
  adaptive: AdaptiveStatus;
  health: HealthStatus;
  quality: QualityStatus;
  collaboration: CollaborationStatus;
  timestamp: Date;
}

export interface Recommendation {
  category: string;
  priority: 'high' | 'medium' | 'low';
  description: string;
  action: string;
}

// Example usage demonstration
export async function demonstrateFramework(): Promise<void> {
  console.log('🚀 SynergyXintrextro Framework Demo');
  console.log('Embodying: "Code is capacity: to act, adapt, and evolve"');
  
  const framework = new SynergyFramework();
  
  // Process some sample data
  const sampleData = { value: 42, type: 'test', timestamp: new Date() };
  const result = await framework.process(sampleData);
  
  console.log('\n📊 Processing Result:', {
    success: result.success,
    processingTime: `${result.processingTime}ms`,
    qualityScore: result.qualityResult.overallScore
  });
  
  // Show system status
  const status = framework.getSystemStatus();
  console.log('\n🔍 System Status:', {
    adaptiveStrategies: status.adaptive.totalStrategies,
    systemHealth: status.health.isHealthy ? '✅ Healthy' : '⚠️ Issues',
    qualityTrend: status.quality.trend
  });
  
  // Get recommendations
  const recommendations = framework.getRecommendations();
  if (recommendations.length > 0) {
    console.log('\n💡 Recommendations:');
    recommendations.forEach(r => 
      console.log(`  ${r.priority.toUpperCase()}: ${r.description}`)
    );
  }
  
  console.log('\n✨ Framework demonstrates disciplined evolution in action!');
}