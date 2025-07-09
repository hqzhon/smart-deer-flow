# DeerFlow Project Rules

## Project Overview
DeerFlow is an AI-powered deep research framework with multi-agent collaboration, featuring a Python backend (FastAPI, LangChain, LangGraph) and Next.js frontend (React 19, TypeScript, Tailwind CSS).

## Code Style & Formatting

### Python Backend
- Use **Black** for code formatting (line length: 88)
- Follow **PEP 8** naming conventions
- Use **type hints** for all function parameters and return values
- Prefer **async/await** for I/O operations
- Use **dataclasses** or **Pydantic models** for data structures
- Import order: standard library, third-party, local imports
- Use **f-strings** for string formatting
- Maximum line length: 88 characters
- Use **docstrings** for all public functions and classes

### Frontend (React/TypeScript)
- Use **Prettier** for code formatting
- Follow **camelCase** for variables and functions
- Use **PascalCase** for components and types
- Prefer **arrow functions** for components
- Use **TypeScript strict mode**
- Prefer **const assertions** and **as const**
- Use **interface** over **type** for object shapes
- Organize imports: React, third-party, local components, utilities

## Architecture Patterns

### Backend Architecture
- Follow **Clean Architecture** principles
- Use **dependency injection** for services
- Implement **repository pattern** for data access
- Use **factory pattern** for agent creation
- Apply **strategy pattern** for different search engines
- Implement **observer pattern** for event handling
- Use **command pattern** for agent actions

### Frontend Architecture
- Use **component composition** over inheritance
- Implement **custom hooks** for reusable logic
- Use **Zustand** for global state management
- Apply **compound component pattern** for complex UI
- Use **render props** or **children as function** when needed
- Implement **error boundaries** for error handling

## File Organization

### Backend Structure
```
src/
├── agents/          # Multi-agent system components
├── api/             # FastAPI routes and endpoints
├── core/            # Core business logic
├── models/          # Pydantic models and data structures
├── services/        # External service integrations
├── tools/           # Agent tools and utilities
├── utils/           # Helper functions and utilities
└── config/          # Configuration management
```

### Frontend Structure
```
web/
├── src/
│   ├── components/  # Reusable UI components
│   ├── pages/       # Next.js pages
│   ├── hooks/       # Custom React hooks
│   ├── stores/      # Zustand stores
│   ├── types/       # TypeScript type definitions
│   ├── utils/       # Utility functions
│   └── styles/      # Global styles and Tailwind config
```

## Naming Conventions

### Python
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Agent classes**: `{Purpose}Agent` (e.g., `ResearchAgent`)
- **Service classes**: `{Service}Service` (e.g., `TavilyService`)

### TypeScript/React
- **Files**: `kebab-case.tsx` for components, `camelCase.ts` for utilities
- **Components**: `PascalCase`
- **Hooks**: `use{Purpose}` (e.g., `useResearch`)
- **Types/Interfaces**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Props interfaces**: `{Component}Props`

## Error Handling

### Python
- Use **custom exception classes** for domain-specific errors
- Implement **structured logging** with context
- Use **try-except-finally** blocks appropriately
- Log errors with **correlation IDs** for tracing
- Implement **circuit breaker pattern** for external services
- Use **exponential backoff** for retries

### Frontend
- Implement **error boundaries** for component trees
- Use **React Query** for API error handling
- Display **user-friendly error messages**
- Log errors to **monitoring service**
- Implement **fallback UI** for failed components

## Performance Guidelines

### Python
- Use **async/await** for I/O-bound operations
- Implement **connection pooling** for databases
- Use **caching strategies** (Redis, in-memory)
- Apply **lazy loading** for heavy resources
- Implement **rate limiting** for API endpoints
- Use **background tasks** for long-running operations

### Frontend
- Use **React.memo** for expensive components
- Implement **code splitting** with dynamic imports
- Use **useMemo** and **useCallback** judiciously
- Optimize **bundle size** with tree shaking
- Implement **virtual scrolling** for large lists
- Use **Suspense** for data fetching

## Testing Standards

### Python Testing
- Use **pytest** for all tests
- Maintain **>80% code coverage**
- Write **unit tests** for business logic
- Implement **integration tests** for APIs
- Use **fixtures** for test data
- Mock **external dependencies**
- Test **error scenarios** and edge cases

### Frontend Testing
- Use **Jest** and **React Testing Library**
- Test **user interactions** and behavior
- Mock **API calls** and external dependencies
- Test **accessibility** with screen readers
- Implement **visual regression tests**
- Test **responsive design** across devices

## Security Best Practices

### Backend Security
- **Never commit** API keys or secrets
- Use **environment variables** for configuration
- Implement **input validation** and sanitization
- Use **HTTPS** for all communications
- Implement **rate limiting** and **CORS** properly
- Validate **JWT tokens** and implement proper auth
- Use **parameterized queries** to prevent SQL injection

### Frontend Security
- **Sanitize user inputs** to prevent XSS
- Use **Content Security Policy** headers
- Implement **proper authentication** flows
- **Never expose** sensitive data in client code
- Use **HTTPS** for all API calls
- Validate **data from APIs** before using

## Documentation Standards

### Code Documentation
- Write **clear docstrings** for all public APIs
- Include **type hints** in Python code
- Document **complex algorithms** and business logic
- Maintain **API documentation** with examples
- Use **JSDoc** for TypeScript functions
- Document **component props** and usage

### Project Documentation
- Keep **README.md** up to date
- Document **setup and deployment** procedures
- Maintain **API documentation** (OpenAPI/Swagger)
- Document **architecture decisions** (ADRs)
- Provide **troubleshooting guides**
- Include **contribution guidelines**

## Git Workflow

### Commit Messages
- Use **conventional commits** format
- Format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Keep **first line under 50 characters**
- Include **issue numbers** when applicable

### Branch Strategy
- Use **feature branches** for new development
- Branch naming: `feature/description`, `fix/description`
- **Squash commits** before merging
- Require **pull request reviews**
- Run **CI/CD checks** before merging

## AI/LLM Specific Guidelines

### Agent Development
- **Single responsibility** per agent
- Use **clear agent interfaces** and contracts
- Implement **proper state management**
- Handle **agent communication** through well-defined protocols
- Use **structured outputs** for agent responses
- Implement **fallback strategies** for agent failures

### Prompt Engineering
- Store **prompts in separate files** or constants
- Use **template engines** for dynamic prompts
- Implement **prompt versioning** and A/B testing
- **Validate LLM outputs** before processing
- Handle **rate limits** and **token limits** gracefully
- Log **prompt-response pairs** for debugging

### Tool Integration
- **Abstract tool interfaces** for easy swapping
- Implement **tool result validation**
- Handle **tool failures** gracefully
- Use **structured schemas** for tool inputs/outputs
- Implement **tool usage analytics**

## Dependencies Management

### Python
- Use **pyproject.toml** for dependency management
- Pin **major versions** in production
- Regular **security audits** with `pip-audit`
- Use **virtual environments** for isolation
- Document **dependency rationale**

### Node.js
- Use **pnpm** for package management
- Lock **exact versions** in package-lock
- Regular **vulnerability scans** with `npm audit`
- Use **peer dependencies** appropriately
- Minimize **bundle size** impact

## Environment Configuration

### Development
- Use **.env files** for local development
- Provide **.env.example** templates
- **Never commit** actual .env files
- Use **different configs** for different environments
- Implement **config validation** on startup

### Production
- Use **environment variables** for configuration
- Implement **health checks** and monitoring
- Use **structured logging** with correlation IDs
- Implement **graceful shutdown** procedures
- Monitor **performance metrics** and errors

## Code Review Guidelines

### Review Checklist
- **Functionality**: Does the code work as intended?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security vulnerabilities?
- **Maintainability**: Is the code readable and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code properly documented?

### Review Process
- **Small, focused** pull requests
- **Self-review** before requesting review
- **Constructive feedback** with suggestions
- **Approve only** when confident in changes
- **Follow up** on requested changes

## Monitoring & Observability

### Logging
- Use **structured logging** (JSON format)
- Include **correlation IDs** for request tracing
- Log **important business events**
- Implement **different log levels** appropriately
- **Never log** sensitive information

### Metrics
- Monitor **API response times** and error rates
- Track **agent performance** and success rates
- Monitor **resource usage** (CPU, memory, disk)
- Implement **custom business metrics**
- Set up **alerting** for critical issues

## Deployment

### CI/CD Pipeline
- **Automated testing** on all commits
- **Code quality checks** (linting, formatting)
- **Security scanning** for vulnerabilities
- **Automated deployment** to staging
- **Manual approval** for production deployment

### Infrastructure
- Use **containerization** (Docker) for consistency
- Implement **blue-green deployments**
- Use **infrastructure as code** (Terraform, etc.)
- Implement **database migrations** safely
- Monitor **deployment health** and rollback if needed

This document should be regularly updated as the project evolves and new patterns emerge.