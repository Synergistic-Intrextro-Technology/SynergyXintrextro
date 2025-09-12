/**
 * Collaboration System
 * Implements: "Encourage collaboration, transparency, and self-reflection"
 */
export interface CollaborationEvent {
  id: string;
  type: 'code-review' | 'knowledge-share' | 'reflection' | 'decision';
  timestamp: Date;
  participants: string[];
  content: string;
  metadata?: Record<string, any>;
}

export interface KnowledgeItem {
  id: string;
  title: string;
  content: string;
  author: string;
  tags: string[];
  created: Date;
  lastUpdated: Date;
  references?: string[];
}

export interface ReflectionPrompt {
  question: string;
  category: 'technical' | 'process' | 'quality' | 'learning';
  frequency: 'daily' | 'weekly' | 'sprint' | 'milestone';
}

/**
 * Collaboration Hub
 * Facilitates: "transparency" and "collaboration"
 */
export class CollaborationHub {
  private events: CollaborationEvent[] = [];
  private knowledge: Map<string, KnowledgeItem> = new Map();
  private reflectionPrompts: ReflectionPrompt[] = [];

  constructor() {
    this.initializeDefaultPrompts();
  }

  /**
   * Record a collaboration event
   * Supports: "transparency" in team interactions
   */
  recordEvent(event: Omit<CollaborationEvent, 'id' | 'timestamp'>): string {
    const collaborationEvent: CollaborationEvent = {
      ...event,
      id: this.generateId(),
      timestamp: new Date()
    };

    this.events.push(collaborationEvent);
    return collaborationEvent.id;
  }

  /**
   * Add knowledge to shared repository
   * Enables: "collaboration" and knowledge sharing
   */
  addKnowledge(knowledge: Omit<KnowledgeItem, 'id' | 'created' | 'lastUpdated'>): string {
    const item: KnowledgeItem = {
      ...knowledge,
      id: this.generateId(),
      created: new Date(),
      lastUpdated: new Date()
    };

    this.knowledge.set(item.id, item);
    return item.id;
  }

  /**
   * Update existing knowledge
   * Demonstrates: "systems live, experiment, and balance stability with novelty"
   */
  updateKnowledge(id: string, updates: Partial<Omit<KnowledgeItem, 'id' | 'created'>>): boolean {
    const existing = this.knowledge.get(id);
    if (!existing) return false;

    const updated: KnowledgeItem = {
      ...existing,
      ...updates,
      lastUpdated: new Date()
    };

    this.knowledge.set(id, updated);
    return true;
  }

  /**
   * Search knowledge base
   * Facilitates: "collaboration" through knowledge discovery
   */
  searchKnowledge(query: string, tags?: string[]): KnowledgeItem[] {
    const results: KnowledgeItem[] = [];
    const queryLower = query.toLowerCase();

    for (const item of this.knowledge.values()) {
      let matches = false;

      // Text search
      if (item.title.toLowerCase().includes(queryLower) ||
          item.content.toLowerCase().includes(queryLower)) {
        matches = true;
      }

      // Tag filter
      if (tags && tags.length > 0) {
        const hasAllTags = tags.every(tag => 
          item.tags.some(itemTag => itemTag.toLowerCase() === tag.toLowerCase())
        );
        matches = matches && hasAllTags;
      }

      if (matches) {
        results.push(item);
      }
    }

    return results.sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime());
  }

  /**
   * Get reflection prompts for team growth
   * Promotes: "self-reflection" and continuous improvement
   */
  getReflectionPrompts(category?: ReflectionPrompt['category']): ReflectionPrompt[] {
    if (!category) return [...this.reflectionPrompts];
    return this.reflectionPrompts.filter(p => p.category === category);
  }

  /**
   * Generate collaboration insights
   * Supports: "transparency" and team analytics
   */
  getCollaborationInsights(days: number = 30): CollaborationInsights {
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    const recentEvents = this.events.filter(e => e.timestamp >= cutoff);

    const eventsByType = new Map<string, number>();
    const participantActivity = new Map<string, number>();

    for (const event of recentEvents) {
      // Count events by type
      eventsByType.set(event.type, (eventsByType.get(event.type) || 0) + 1);

      // Count participant activity
      for (const participant of event.participants) {
        participantActivity.set(participant, (participantActivity.get(participant) || 0) + 1);
      }
    }

    return {
      totalEvents: recentEvents.length,
      eventsByType: Object.fromEntries(eventsByType),
      activeParticipants: participantActivity.size,
      participantActivity: Object.fromEntries(participantActivity),
      knowledgeItems: this.knowledge.size,
      period: `${days} days`
    };
  }

  /**
   * Facilitate code review process
   * Embodies: "collaboration" and "quality" focus
   */
  initiateCodeReview(code: string, author: string, reviewers: string[]): CodeReviewSession {
    const sessionId = this.generateId();
    
    // Record the review initiation
    this.recordEvent({
      type: 'code-review',
      participants: [author, ...reviewers],
      content: `Code review initiated for ${code.length} characters of code`,
      metadata: {
        sessionId,
        author,
        reviewers,
        codeLength: code.length
      }
    });

    return {
      sessionId,
      code,
      author,
      reviewers,
      comments: [],
      status: 'pending',
      created: new Date()
    };
  }

  private initializeDefaultPrompts(): void {
    this.reflectionPrompts = [
      {
        question: "What did we learn from today's challenges?",
        category: 'learning',
        frequency: 'daily'
      },
      {
        question: "How can we improve our code quality practices?",
        category: 'quality',
        frequency: 'weekly'
      },
      {
        question: "What processes are helping or hindering our progress?",
        category: 'process',
        frequency: 'sprint'
      },
      {
        question: "What technical decisions need revisiting?",
        category: 'technical',
        frequency: 'milestone'
      },
      {
        question: "How well are we balancing stability with innovation?",
        category: 'technical',
        frequency: 'sprint'
      }
    ];
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  }
}

export interface CollaborationInsights {
  totalEvents: number;
  eventsByType: Record<string, number>;
  activeParticipants: number;
  participantActivity: Record<string, number>;
  knowledgeItems: number;
  period: string;
}

export interface CodeReviewSession {
  sessionId: string;
  code: string;
  author: string;
  reviewers: string[];
  comments: ReviewComment[];
  status: 'pending' | 'in-review' | 'approved' | 'rejected';
  created: Date;
}

export interface ReviewComment {
  author: string;
  content: string;
  line?: number;
  timestamp: Date;
  type: 'suggestion' | 'issue' | 'approval';
}