# -*- coding: utf-8 -*-
"""
Dependency Injection Container Implementation
Provides IoC container functionality to reduce hard-coded dependencies and improve code testability and maintainability
"""

import logging
import inspect
from typing import Any, Dict, Type, TypeVar, Callable, Optional, Union, get_type_hints
from functools import wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LifecycleScope(Enum):
    """Lifecycle Scope"""
    SINGLETON = "singleton"    # Singleton
    TRANSIENT = "transient"    # Transient
    SCOPED = "scoped"          # Scoped


@dataclass
class ServiceDescriptor:
    """Service Descriptor"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifecycle: LifecycleScope = LifecycleScope.TRANSIENT
    dependencies: Optional[Dict[str, Type]] = None


class DependencyInjectionContainer:
    """Dependency Injection Container"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._building_stack: set = set()  # Prevent circular dependencies
    
    def register_singleton(self, service_type: Type[T], implementation_type: Type[T] = None, factory: Callable[[], T] = None, instance: T = None) -> 'DependencyInjectionContainer':
        """Register singleton service"""
        return self._register_service(service_type, implementation_type, factory, instance, LifecycleScope.SINGLETON)
    
    def register_transient(self, service_type: Type[T], implementation_type: Type[T] = None, factory: Callable[[], T] = None) -> 'DependencyInjectionContainer':
        """Register transient service"""
        return self._register_service(service_type, implementation_type, factory, None, LifecycleScope.TRANSIENT)
    
    def register_scoped(self, service_type: Type[T], implementation_type: Type[T] = None, factory: Callable[[], T] = None) -> 'DependencyInjectionContainer':
        """Register scoped service"""
        return self._register_service(service_type, implementation_type, factory, None, LifecycleScope.SCOPED)
    
    def _register_service(self, service_type: Type, implementation_type: Type = None, factory: Callable = None, instance: Any = None, lifecycle: LifecycleScope = LifecycleScope.TRANSIENT) -> 'DependencyInjectionContainer':
        """Register service"""
        if sum(x is not None for x in [implementation_type, factory, instance]) != 1:
            raise ValueError("Must provide exactly one of: implementation_type, factory, or instance")
        
        # Analyze dependencies
        dependencies = None
        if implementation_type:
            dependencies = self._analyze_dependencies(implementation_type)
        elif factory:
            dependencies = self._analyze_dependencies(factory)
        
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifecycle=lifecycle,
            dependencies=dependencies
        )
        
        self._services[service_type] = descriptor
        
        # If singleton and instance provided, store directly
        if lifecycle == LifecycleScope.SINGLETON and instance is not None:
            self._instances[service_type] = instance
        
        logger.debug(f"Registered {service_type.__name__} as {lifecycle.value}")
        return self
    
    def _analyze_dependencies(self, target: Union[Type, Callable]) -> Dict[str, Type]:
        """Analyze dependencies"""
        dependencies = {}
        
        try:
            if inspect.isclass(target):
                # Analyze constructor
                init_method = target.__init__
                sig = inspect.signature(init_method)
                type_hints = get_type_hints(init_method)
            else:
                # Analyze function
                sig = inspect.signature(target)
                type_hints = get_type_hints(target)
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = type_hints.get(param_name)
                if param_type:
                    dependencies[param_name] = param_type
                elif param.annotation != inspect.Parameter.empty:
                    dependencies[param_name] = param.annotation
        
        except Exception as e:
            logger.warning(f"Failed to analyze dependencies for {target}: {e}")
        
        return dependencies
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service"""
        if service_type in self._building_stack:
            raise RuntimeError(f"Circular dependency detected for {service_type.__name__}")
        
        descriptor = self._services.get(service_type)
        if not descriptor:
            raise ValueError(f"Service {service_type.__name__} is not registered")
        
        # Check lifecycle
        if descriptor.lifecycle == LifecycleScope.SINGLETON:
            if service_type in self._instances:
                return self._instances[service_type]
        elif descriptor.lifecycle == LifecycleScope.SCOPED:
            if self._current_scope and self._current_scope in self._scoped_instances:
                scoped_instances = self._scoped_instances[self._current_scope]
                if service_type in scoped_instances:
                    return scoped_instances[service_type]
        
        # Create instance
        self._building_stack.add(service_type)
        try:
            instance = self._create_instance(descriptor)
            
            # Store instance
            if descriptor.lifecycle == LifecycleScope.SINGLETON:
                self._instances[service_type] = instance
            elif descriptor.lifecycle == LifecycleScope.SCOPED and self._current_scope:
                if self._current_scope not in self._scoped_instances:
                    self._scoped_instances[self._current_scope] = {}
                self._scoped_instances[self._current_scope][service_type] = instance
            
            return instance
        finally:
            self._building_stack.remove(service_type)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create instance"""
        if descriptor.instance is not None:
            return descriptor.instance
        
        if descriptor.factory:
            # Use factory method
            if descriptor.dependencies:
                kwargs = self._resolve_dependencies(descriptor.dependencies)
                return descriptor.factory(**kwargs)
            else:
                return descriptor.factory()
        
        if descriptor.implementation_type:
            # Use implementation type
            if descriptor.dependencies:
                kwargs = self._resolve_dependencies(descriptor.dependencies)
                return descriptor.implementation_type(**kwargs)
            else:
                return descriptor.implementation_type()
        
        raise ValueError(f"Cannot create instance for {descriptor.service_type.__name__}")
    
    def _resolve_dependencies(self, dependencies: Dict[str, Type]) -> Dict[str, Any]:
        """Resolve dependencies"""
        resolved = {}
        for param_name, param_type in dependencies.items():
            try:
                resolved[param_name] = self.resolve(param_type)
            except Exception as e:
                logger.warning(f"Failed to resolve dependency {param_name} of type {param_type.__name__}: {e}")
                # Can choose to skip optional dependencies or throw exception
                raise
        return resolved
    
    def create_scope(self, scope_name: str = None) -> 'ScopeContext':
        """Create scope"""
        if scope_name is None:
            import uuid
            scope_name = str(uuid.uuid4())
        
        return ScopeContext(self, scope_name)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        return service_type in self._services
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services"""
        return self._services.copy()
    
    def clear(self):
        """Clear all registered services"""
        self._services.clear()
        self._instances.clear()
        self._scoped_instances.clear()
        self._current_scope = None
        self._building_stack.clear()


class ScopeContext:
    """Scope Context"""
    
    def __init__(self, container: DependencyInjectionContainer, scope_name: str):
        self.container = container
        self.scope_name = scope_name
        self.previous_scope = None
    
    def __enter__(self):
        self.previous_scope = self.container._current_scope
        self.container._current_scope = self.scope_name
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up scoped instances
        if self.scope_name in self.container._scoped_instances:
            del self.container._scoped_instances[self.scope_name]
        
        self.container._current_scope = self.previous_scope


# Global container instance
global_container = DependencyInjectionContainer()


def inject(*dependencies: Type):
    """Dependency injection decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve dependencies and inject
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            injected_kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name in kwargs:
                    continue  # Parameter already provided
                
                param_type = type_hints.get(param_name)
                if param_type and global_container.is_registered(param_type):
                    injected_kwargs[param_name] = global_container.resolve(param_type)
            
            return func(*args, **kwargs, **injected_kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Resolve dependencies and inject
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            injected_kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name in kwargs:
                    continue  # Parameter already provided
                
                param_type = type_hints.get(param_name)
                if param_type and global_container.is_registered(param_type):
                    injected_kwargs[param_name] = global_container.resolve(param_type)
            
            return await func(*args, **kwargs, **injected_kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def configure_services(configuration_func: Callable[[DependencyInjectionContainer], None]):
    """Configure services"""
    configuration_func(global_container)


# Common service interfaces
class ILogger:
    """Logger interface"""
    def debug(self, message: str): pass
    def info(self, message: str): pass
    def warning(self, message: str): pass
    def error(self, message: str): pass


class IConfigurationService:
    """Configuration service interface"""
    def get(self, key: str, default=None): pass
    def set(self, key: str, value): pass


class ILLMService:
    """LLM service interface"""
    async def invoke(self, messages, **kwargs): pass


class IRateLimiter:
    """Rate limiter interface"""
    async def acquire(self) -> float: pass
    def record_success(self): pass
    def record_failure(self, error: str): pass