// SPDX-License-Identifier: MIT

/**
 * Agent configuration for dynamic agent type handling
 */
export interface AgentConfig {
  /** Agent type identifier */
  type: string;
  /** Display name for the agent */
  displayName: string;
  /** Whether this agent should be included in research activities */
  includeInResearch: boolean;
  /** Whether this agent should be displayed in activity messages */
  displayInActivity: boolean;
  /** Color theme for the agent (optional) */
  color?: string;
  /** Icon for the agent (optional) */
  icon?: string;
}

/**
 * Default agent configurations
 * This can be extended or modified without changing core logic
 */
export const DEFAULT_AGENT_CONFIGS: AgentConfig[] = [
  {
    type: "researcher",
    displayName: "Researcher",
    includeInResearch: true,
    displayInActivity: true,
    color: "blue",
    icon: "🔍"
  },
  {
    type: "coder",
    displayName: "Coder",
    includeInResearch: true,
    displayInActivity: true,
    color: "green",
    icon: "💻"
  },
  {
    type: "reporter",
    displayName: "Reporter",
    includeInResearch: true,
    displayInActivity: false, // Excluded from activity display as per original logic
    color: "purple",
    icon: "📝"
  },
  {
    type: "planner",
    displayName: "Planner",
    includeInResearch: false,
    displayInActivity: false, // Excluded from activity display as per original logic
    color: "orange",
    icon: "📋"
  },
  {
    type: "research_team",
    displayName: "Research Team",
    includeInResearch: true,
    displayInActivity: true,
    color: "indigo",
    icon: "👥"
  },
  {
    type: "web_search",
    displayName: "Web Search",
    includeInResearch: true,
    displayInActivity: true,
    color: "indigo",
    icon: "🔍"
  }
];

/**
 * Agent configuration manager
 */
export class AgentConfigManager {
  private static configs = new Map<string, AgentConfig>();
  private static initialized = false;

  /**
   * Initialize the agent configuration manager
   */
  static initialize(configs: AgentConfig[] = DEFAULT_AGENT_CONFIGS) {
    this.configs.clear();
    configs.forEach(config => {
      this.configs.set(config.type, config);
    });
    this.initialized = true;
  }

  /**
   * Get configuration for a specific agent type
   */
  static getConfig(agentType: string): AgentConfig | undefined {
    if (!this.initialized) {
      this.initialize();
    }
    return this.configs.get(agentType);
  }

  /**
   * Check if an agent should be included in research activities
   */
  static shouldIncludeInResearch(agentType: string): boolean {
    const config = this.getConfig(agentType);
    return config?.includeInResearch ?? false;
  }

  /**
   * Check if an agent should be displayed in activity messages
   */
  static shouldDisplayInActivity(agentType: string): boolean {
    const config = this.getConfig(agentType);
    return config?.displayInActivity ?? false;
  }

  /**
   * Get all agent types that should be included in research
   */
  static getResearchAgentTypes(): string[] {
    if (!this.initialized) {
      this.initialize();
    }
    return Array.from(this.configs.values())
      .filter(config => config.includeInResearch)
      .map(config => config.type);
  }

  /**
   * Check if the agent is a reporter
   */
  static isReporter(agentType: string): boolean {
    return agentType === 'reporter';
  }

  /**
   * Get all agent types that should be displayed in activities
   */
  static getActivityAgentTypes(): string[] {
    if (!this.initialized) {
      this.initialize();
    }
    return Array.from(this.configs.values())
      .filter(config => config.displayInActivity)
      .map(config => config.type);
  }

  /**
   * Add or update an agent configuration
   */
  static addConfig(config: AgentConfig) {
    if (!this.initialized) {
      this.initialize();
    }
    this.configs.set(config.type, config);
  }

  /**
   * Remove an agent configuration
   */
  static removeConfig(agentType: string) {
    this.configs.delete(agentType);
  }

  /**
   * Get all configured agent types
   */
  static getAllAgentTypes(): string[] {
    if (!this.initialized) {
      this.initialize();
    }
    return Array.from(this.configs.keys());
  }

  /**
   * Get all agent configurations
   */
  static getAllConfigs(): AgentConfig[] {
    if (!this.initialized) {
      this.initialize();
    }
    return Array.from(this.configs.values());
  }
}

// Initialize the manager with default configs
AgentConfigManager.initialize();