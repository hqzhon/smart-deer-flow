// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import * as React from "react";
import { cn } from "~/lib/utils";

interface LazyLoadProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  rootMargin?: string;
  threshold?: number;
  className?: string;
  once?: boolean;
}

function LazyLoad({
  children,
  fallback = null,
  rootMargin = "100px",
  threshold = 0.1,
  className,
  once = true,
}: LazyLoadProps) {
  const [isVisible, setIsVisible] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry && entry.isIntersecting) {
          setIsVisible(true);
          if (once) {
            observer.unobserve(element);
          }
        } else if (!once && entry) {
          setIsVisible(false);
        }
      },
      { rootMargin, threshold }
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, [rootMargin, threshold, once]);

  return (
    <div ref={ref} className={className}>
      {isVisible ? children : fallback}
    </div>
  );
}

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number;
  containerHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  overscan?: number;
  className?: string;
  onEndReached?: () => void;
  endReachedThreshold?: number;
}

function VirtualList<T>({
  items,
  itemHeight,
  containerHeight,
  renderItem,
  overscan = 3,
  className,
  onEndReached,
  endReachedThreshold = 0.8,
}: VirtualListProps<T>) {
  const [scrollTop, setScrollTop] = React.useState(0);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const totalHeight = items.length * itemHeight;
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(
    items.length,
    Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
  );

  const visibleItems = items.slice(startIndex, endIndex);

  React.useEffect(() => {
    if (onEndReached && endIndex / items.length >= endReachedThreshold) {
      onEndReached();
    }
  }, [endIndex, items.length, onEndReached, endReachedThreshold]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  };

  return (
    <div
      ref={containerRef}
      className={cn("overflow-auto", className)}
      style={{ height: containerHeight }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: "relative" }}>
        {visibleItems.map((item, index) => (
          <div
            key={startIndex + index}
            style={{
              position: "absolute",
              top: (startIndex + index) * itemHeight,
              height: itemHeight,
              width: "100%",
            }}
          >
            {renderItem(item, startIndex + index)}
          </div>
        ))}
      </div>
    </div>
  );
}

interface DataCacheOptions<T> {
  key: string;
  fetcher: () => Promise<T>;
  staleTime?: number;
  cacheTime?: number;
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

const globalCache = new Map<string, CacheEntry<unknown>>();

function useDataCache<T>({
  key,
  fetcher,
  staleTime = 5 * 60 * 1000,
  cacheTime = 30 * 60 * 1000,
}: DataCacheOptions<T>) {
  const [data, setData] = React.useState<T | null>(null);
  const [error, setError] = React.useState<Error | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  const fetchData = React.useCallback(async () => {
    const cached = globalCache.get(key) as CacheEntry<T> | undefined;
    const now = Date.now();

    if (cached && now - cached.timestamp < staleTime) {
      setData(cached.data);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await fetcher();
      setData(result);
      globalCache.set(key, { data: result, timestamp: now });
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  }, [key, fetcher, staleTime]);

  const invalidate = React.useCallback(() => {
    globalCache.delete(key);
    fetchData();
  }, [key, fetchData]);

  const clear = React.useCallback(() => {
    globalCache.delete(key);
    setData(null);
    setError(null);
  }, [key]);

  React.useEffect(() => {
    fetchData();

    const interval = setInterval(() => {
      const cached = globalCache.get(key);
      if (cached && Date.now() - cached.timestamp > cacheTime) {
        globalCache.delete(key);
      }
    }, cacheTime);

    return () => clearInterval(interval);
  }, [fetchData, key, cacheTime]);

  return { data, error, isLoading, invalidate, clear, refetch: fetchData };
}

interface DebounceOptions {
  delay?: number;
  leading?: boolean;
  trailing?: boolean;
}

function useDebounce<T>(value: T, options: DebounceOptions = {}): T {
  const { delay = 300, leading = false, trailing = true } = options;
  const [debouncedValue, setDebouncedValue] = React.useState(value);
  const isFirstCall = React.useRef(true);

  React.useEffect(() => {
    if (isFirstCall.current && leading) {
      setDebouncedValue(value);
      isFirstCall.current = false;
      return;
    }

    const timer = setTimeout(() => {
      if (trailing) {
        setDebouncedValue(value);
      }
    }, delay);

    return () => clearTimeout(timer);
  }, [value, delay, leading, trailing]);

  return debouncedValue;
}

function useThrottle<T>(value: T, interval: number = 300): T {
  const [throttledValue, setThrottledValue] = React.useState(value);
  const lastUpdated = React.useRef(Date.now());

  React.useEffect(() => {
    const now = Date.now();
    if (now - lastUpdated.current >= interval) {
      lastUpdated.current = now;
      setThrottledValue(value);
    } else {
      const timer = setTimeout(() => {
        lastUpdated.current = Date.now();
        setThrottledValue(value);
      }, interval - (now - lastUpdated.current));

      return () => clearTimeout(timer);
    }
  }, [value, interval]);

  return throttledValue;
}

interface ImageOptimizationProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  placeholder?: string;
  blur?: boolean;
}

function OptimizedImage({
  src,
  alt,
  width,
  height,
  className,
  placeholder = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect fill='%23f1f5f9' width='400' height='300'/%3E%3C/svg%3E",
  blur = true,
}: ImageOptimizationProps) {
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [hasError, setHasError] = React.useState(false);

  return (
    <div
      className={cn("relative overflow-hidden", className)}
      style={{ width, height }}
    >
      {blur && !isLoaded && (
        <img
          src={placeholder}
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
        />
      )}
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        loading="lazy"
        className={cn(
          "transition-opacity duration-300",
          blur && !isLoaded && "opacity-0",
          isLoaded && "opacity-100",
          hasError && "hidden"
        )}
        onLoad={() => setIsLoaded(true)}
        onError={() => setHasError(true)}
      />
      {hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted text-muted-foreground text-sm">
          Failed to load image
        </div>
      )}
    </div>
  );
}

interface PerformanceMonitorProps {
  children: React.ReactNode;
  onRender?: (duration: number) => void;
  name?: string;
}

function PerformanceMonitor({
  children,
  onRender,
  name = "Component",
}: PerformanceMonitorProps) {
  const renderCount = React.useRef(0);

  React.useEffect(() => {
    renderCount.current += 1;
    const startTime = performance.now();

    return () => {
      const duration = performance.now() - startTime;
      if (onRender) {
        onRender(duration);
      }
      if (process.env.NODE_ENV === "development") {
        console.log(
          `[${name}] Render #${renderCount.current} took ${duration.toFixed(2)}ms`
        );
      }
    };
  });

  return <>{children}</>;
}

interface MemoizedListProps<T> {
  items: T[];
  keyExtractor: (item: T) => string;
  renderItem: (item: T) => React.ReactNode;
  className?: string;
}

function MemoizedList<T>({
  items,
  keyExtractor,
  renderItem,
  className,
}: MemoizedListProps<T>) {
  return (
    <div className={className}>
      {items.map((item, index) => (
        <React.Fragment key={keyExtractor(item)}>
          {renderItem(item)}
        </React.Fragment>
      ))}
    </div>
  );
}

export {
  LazyLoad,
  VirtualList,
  useDataCache,
  useDebounce,
  useThrottle,
  OptimizedImage,
  PerformanceMonitor,
  MemoizedList,
};
