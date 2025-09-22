/**
 * Quality Control System
 * Implements: "Build only what strengthens control, stability, and quality"
 */
export interface QualityCheck {
  name: string;
  description: string;
  execute<T>(input: T): Promise<QualityResult>;
}

export interface QualityResult {
  passed: boolean;
  score: number; // 0-100
  message: string;
  suggestions?: string[];
  metrics?: Record<string, number>;
}

export interface QualityGate {
  name: string;
  checks: QualityCheck[];
  minimumScore: number;
  required: boolean;
}

/**
 * Quality Gate System
 * Ensures: "stability" and "quality" in all operations
 */
export class QualityController {
  private gates: Map<string, QualityGate> = new Map();
  private history: QualityHistory[] = [];

  /**
   * Register a quality gate
   * Supports: "Engineer for clarity, security, and maintainability"
   */
  registerGate(gate: QualityGate): void {
    this.gates.set(gate.name, gate);
  }

  /**
   * Execute quality checks through specified gate
   * Enforces: "Build only what strengthens control, stability, and quality"
   */
  async executeGate<T>(gateName: string, input: T): Promise<GateResult> {
    const gate = this.gates.get(gateName);
    if (!gate) {
      throw new Error(`Quality gate '${gateName}' not found`);
    }

    const startTime = Date.now();
    const results: QualityResult[] = [];
    let overallScore = 0;
    let allPassed = true;

    for (const check of gate.checks) {
      try {
        const result = await check.execute(input);
        results.push(result);
        overallScore += result.score;
        
        if (!result.passed) {
          allPassed = false;
        }
      } catch (error) {
        const failedResult: QualityResult = {
          passed: false,
          score: 0,
          message: `Check '${check.name}' failed with error: ${error instanceof Error ? error.message : String(error)}`,
          suggestions: ['Review check implementation', 'Verify input data']
        };
        results.push(failedResult);
        allPassed = false;
      }
    }

    const averageScore = gate.checks.length > 0 ? overallScore / gate.checks.length : 0;
    const meetsMinimum = averageScore >= gate.minimumScore;
    const gateResult: GateResult = {
      gateName,
      passed: allPassed && meetsMinimum,
      overallScore: averageScore,
      results,
      executionTime: Date.now() - startTime,
      timestamp: new Date()
    };

    // Record for learning and improvement
    this.recordHistory(gateResult);

    return gateResult;
  }

  /**
   * Get quality trends for system improvement
   * Enables: "performance-based learning" and "self-reflection"
   */
  getQualityTrends(gateName?: string, days: number = 7): QualityTrends {
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    const relevantHistory = this.history.filter(h => {
      return h.timestamp >= cutoff && (!gateName || h.gateName === gateName);
    });

    if (relevantHistory.length === 0) {
      return {
        totalExecutions: 0,
        passRate: 0,
        averageScore: 0,
        trend: 'no-data'
      };
    }

    const passRate = relevantHistory.filter(h => h.passed).length / relevantHistory.length;
    const averageScore = relevantHistory.reduce((sum, h) => sum + h.overallScore, 0) / relevantHistory.length;
    
    // Calculate trend by comparing first and second half
    const mid = Math.floor(relevantHistory.length / 2);
    const firstHalf = relevantHistory.slice(0, mid);
    const secondHalf = relevantHistory.slice(mid);
    
    const firstHalfAvg = firstHalf.length > 0 
      ? firstHalf.reduce((sum, h) => sum + h.overallScore, 0) / firstHalf.length 
      : 0;
    const secondHalfAvg = secondHalf.length > 0 
      ? secondHalf.reduce((sum, h) => sum + h.overallScore, 0) / secondHalf.length 
      : 0;

    let trend: 'improving' | 'declining' | 'stable';
    const difference = secondHalfAvg - firstHalfAvg;
    if (Math.abs(difference) < 5) {
      trend = 'stable';
    } else if (difference > 0) {
      trend = 'improving';
    } else {
      trend = 'declining';
    }

    return {
      totalExecutions: relevantHistory.length,
      passRate,
      averageScore,
      trend
    };
  }

  /**
   * Suggest improvements based on quality history
   * Demonstrates: "True inventiveness is disciplined evolution"
   */
  getImprovementSuggestions(): ImprovementSuggestion[] {
    const suggestions: ImprovementSuggestion[] = [];
    const recentFailures = this.history
      .filter(h => !h.passed)
      .slice(-20); // Last 20 failures

    // Analyze common failure patterns
    const failurePatterns = new Map<string, number>();
    for (const failure of recentFailures) {
      for (const result of failure.results) {
        if (!result.passed) {
          const key = `${failure.gateName}:${result.message}`;
          failurePatterns.set(key, (failurePatterns.get(key) || 0) + 1);
        }
      }
    }

    // Generate suggestions for frequent failures
    for (const [pattern, count] of failurePatterns) {
      if (count >= 3) { // Threshold for pattern recognition
        suggestions.push({
          priority: count > 5 ? 'high' : 'medium',
          description: `Frequent failure pattern detected: ${pattern}`,
          action: 'Review and strengthen this quality check',
          impact: 'Improved system stability and quality'
        });
      }
    }

    return suggestions;
  }

  private recordHistory(result: GateResult): void {
    this.history.push(result);
    
    // Maintain reasonable history size
    if (this.history.length > 1000) {
      this.history = this.history.slice(-500); // Keep most recent 500
    }
  }
}

export interface GateResult {
  gateName: string;
  passed: boolean;
  overallScore: number;
  results: QualityResult[];
  executionTime: number;
  timestamp: Date;
}

interface QualityHistory extends GateResult {}

export interface QualityTrends {
  totalExecutions: number;
  passRate: number;
  averageScore: number;
  trend: 'improving' | 'declining' | 'stable' | 'no-data';
}

export interface ImprovementSuggestion {
  priority: 'high' | 'medium' | 'low';
  description: string;
  action: string;
  impact: string;
}