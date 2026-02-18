// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import * as React from "react";
import { cn } from "~/lib/utils";

type Breakpoint = "sm" | "md" | "lg" | "xl" | "2xl";

const breakpoints: Record<Breakpoint, number> = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  "2xl": 1536,
};

interface ResponsiveContextValue {
  width: number;
  height: number;
  breakpoint: Breakpoint;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLargeDesktop: boolean;
}

const ResponsiveContext = React.createContext<ResponsiveContextValue | undefined>(
  undefined
);

function useResponsive(): ResponsiveContextValue {
  const context = React.useContext(ResponsiveContext);
  if (!context) {
    throw new Error("useResponsive must be used within a ResponsiveProvider");
  }
  return context;
}

interface ResponsiveProviderProps {
  children: React.ReactNode;
}

function ResponsiveProvider({ children }: ResponsiveProviderProps) {
  const [dimensions, setDimensions] = React.useState({
    width: typeof window !== "undefined" ? window.innerWidth : 1024,
    height: typeof window !== "undefined" ? window.innerHeight : 768,
  });

  React.useEffect(() => {
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const getBreakpoint = (width: number): Breakpoint => {
    if (width < breakpoints.sm) return "sm";
    if (width < breakpoints.md) return "md";
    if (width < breakpoints.lg) return "lg";
    if (width < breakpoints.xl) return "xl";
    return "2xl";
  };

  const value: ResponsiveContextValue = {
    width: dimensions.width,
    height: dimensions.height,
    breakpoint: getBreakpoint(dimensions.width),
    isMobile: dimensions.width < breakpoints.md,
    isTablet: dimensions.width >= breakpoints.md && dimensions.width < breakpoints.lg,
    isDesktop: dimensions.width >= breakpoints.lg && dimensions.width < breakpoints.xl,
    isLargeDesktop: dimensions.width >= breakpoints.xl,
  };

  return (
    <ResponsiveContext.Provider value={value}>
      {children}
    </ResponsiveContext.Provider>
  );
}

interface ResponsiveProps {
  children: React.ReactNode;
  showOn?: Breakpoint[];
  hideOn?: Breakpoint[];
  className?: string;
}

function Responsive({ children, showOn, hideOn, className }: ResponsiveProps) {
  const { breakpoint } = useResponsive();

  if (showOn && !showOn.includes(breakpoint)) {
    return null;
  }

  if (hideOn && hideOn.includes(breakpoint)) {
    return null;
  }

  return <div className={className}>{children}</div>;
}

interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  maxWidth?: "sm" | "md" | "lg" | "xl" | "2xl" | "full" | "none";
  padding?: boolean;
  center?: boolean;
}

function Container({
  children,
  maxWidth = "lg",
  padding = true,
  center = true,
  className,
  ...props
}: ContainerProps) {
  const maxWidthClasses: Record<string, string> = {
    sm: "max-w-screen-sm",
    md: "max-w-screen-md",
    lg: "max-w-screen-lg",
    xl: "max-w-screen-xl",
    "2xl": "max-w-screen-2xl",
    full: "max-w-full",
    none: "",
  };

  return (
    <div
      className={cn(
        "w-full",
        maxWidthClasses[maxWidth],
        padding && "px-4 md:px-6 lg:px-8",
        center && "mx-auto",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  cols?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
  smCols?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
  mdCols?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
  lgCols?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
  gap?: "none" | "sm" | "md" | "lg" | "xl";
}

function Grid({
  children,
  cols = 1,
  smCols,
  mdCols,
  lgCols,
  gap = "md",
  className,
  ...props
}: GridProps) {
  const colClasses: Record<number, string> = {
    1: "grid-cols-1",
    2: "grid-cols-2",
    3: "grid-cols-3",
    4: "grid-cols-4",
    5: "grid-cols-5",
    6: "grid-cols-6",
    12: "grid-cols-12",
  };

  const gapClasses: Record<string, string> = {
    none: "gap-0",
    sm: "gap-2",
    md: "gap-4",
    lg: "gap-6",
    xl: "gap-8",
  };

  return (
    <div
      className={cn(
        "grid",
        colClasses[cols],
        smCols && `sm:grid-cols-${smCols}`,
        mdCols && `md:grid-cols-${mdCols}`,
        lgCols && `lg:grid-cols-${lgCols}`,
        gapClasses[gap],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

interface StackProps extends React.HTMLAttributes<HTMLDivElement> {
  direction?: "horizontal" | "vertical";
  gap?: "none" | "xs" | "sm" | "md" | "lg" | "xl";
  align?: "start" | "center" | "end" | "stretch";
  justify?: "start" | "center" | "end" | "between" | "around";
  wrap?: boolean;
}

function Stack({
  children,
  direction = "vertical",
  gap = "md",
  align = "stretch",
  justify = "start",
  wrap = false,
  className,
  ...props
}: StackProps) {
  const gapClasses: Record<string, string> = {
    none: "gap-0",
    xs: "gap-1",
    sm: "gap-2",
    md: "gap-4",
    lg: "gap-6",
    xl: "gap-8",
  };

  const alignClasses: Record<string, string> = {
    start: "items-start",
    center: "items-center",
    end: "items-end",
    stretch: "items-stretch",
  };

  const justifyClasses: Record<string, string> = {
    start: "justify-start",
    center: "justify-center",
    end: "justify-end",
    between: "justify-between",
    around: "justify-around",
  };

  return (
    <div
      className={cn(
        "flex",
        direction === "vertical" ? "flex-col" : "flex-row",
        gapClasses[gap],
        alignClasses[align],
        justifyClasses[justify],
        wrap && "flex-wrap",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

interface SidebarLayoutProps {
  sidebar: React.ReactNode;
  children: React.ReactNode;
  sidebarWidth?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  className?: string;
}

function SidebarLayout({
  sidebar,
  children,
  sidebarWidth = "280px",
  collapsible = true,
  defaultCollapsed = false,
  className,
}: SidebarLayoutProps) {
  const { isMobile } = useResponsive();
  const [collapsed, setCollapsed] = React.useState(defaultCollapsed || isMobile);

  React.useEffect(() => {
    if (isMobile) {
      setCollapsed(true);
    }
  }, [isMobile]);

  return (
    <div className={cn("flex h-full", className)}>
      <div
        className={cn(
          "shrink-0 border-r transition-all duration-300 overflow-hidden",
          collapsed ? "w-0 md:w-16" : "w-full md:w-auto"
        )}
        style={{ width: !collapsed ? sidebarWidth : undefined }}
      >
        <div className="h-full overflow-auto">{sidebar}</div>
      </div>
      <div className="flex-1 min-w-0 overflow-auto">{children}</div>
      {collapsible && (
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="fixed bottom-4 left-4 z-50 p-2 rounded-full bg-primary text-primary-foreground shadow-lg md:hidden"
        >
          {collapsed ? "→" : "←"}
        </button>
      )}
    </div>
  );
}

interface SplitPaneProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultSplit?: number;
  minSize?: number;
  className?: string;
}

function SplitPane({
  left,
  right,
  defaultSplit = 50,
  minSize = 200,
  className,
}: SplitPaneProps) {
  const { isMobile } = useResponsive();
  const [split, setSplit] = React.useState(defaultSplit);
  const [isDragging, setIsDragging] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const handleMouseDown = () => setIsDragging(true);
  const handleMouseUp = () => setIsDragging(false);

  const handleMouseMove = React.useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const newSplit = ((e.clientX - rect.left) / rect.width) * 100;
      const clampedSplit = Math.max(
        (minSize / rect.width) * 100,
        Math.min(100 - (minSize / rect.width) * 100, newSplit)
      );
      setSplit(clampedSplit);
    },
    [isDragging, minSize]
  );

  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove]);

  if (isMobile) {
    return (
      <div className={cn("flex flex-col h-full", className)}>
        <div className="flex-1 overflow-auto">{left}</div>
        <div className="flex-1 overflow-auto border-t">{right}</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={cn("flex h-full", className)}>
      <div style={{ width: `${split}%` }} className="overflow-auto">
        {left}
      </div>
      <div
        className="w-1 bg-border cursor-col-resize hover:bg-primary/50 transition-colors"
        onMouseDown={handleMouseDown}
      />
      <div style={{ width: `${100 - split}%` }} className="overflow-auto">
        {right}
      </div>
    </div>
  );
}

export {
  ResponsiveProvider,
  useResponsive,
  Responsive,
  Container,
  Grid,
  Stack,
  SidebarLayout,
  SplitPane,
  breakpoints,
};
