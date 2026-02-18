// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import * as React from "react";
import { cn } from "~/lib/utils";

type ModifierKey = "ctrl" | "alt" | "shift" | "meta";

interface Hotkey {
  key: string;
  modifiers?: ModifierKey[];
  description: string;
  action: () => void;
  group?: string;
}

interface HotkeyConfig {
  hotkeys: Hotkey[];
  enabled?: boolean;
  scope?: string;
}

interface HotkeysContextValue {
  registerHotkey: (hotkey: Hotkey) => () => void;
  unregisterHotkey: (key: string, modifiers?: ModifierKey[]) => void;
  getHotkeys: () => Hotkey[];
  isModalOpen: boolean;
  setModalOpen: (open: boolean) => void;
}

const HotkeysContext = React.createContext<HotkeysContextValue | undefined>(
  undefined
);

function useHotkeys(): HotkeysContextValue {
  const context = React.useContext(HotkeysContext);
  if (!context) {
    throw new Error("useHotkeys must be used within a HotkeysProvider");
  }
  return context;
}

interface HotkeysProviderProps {
  children: React.ReactNode;
}

function HotkeysProvider({ children }: HotkeysProviderProps) {
  const hotkeysRef = React.useRef<Hotkey[]>([]);
  const [isModalOpen, setModalOpen] = React.useState(false);

  const registerHotkey = React.useCallback((hotkey: Hotkey) => {
    hotkeysRef.current.push(hotkey);
    return () => {
      hotkeysRef.current = hotkeysRef.current.filter(
        (h) =>
          h.key !== hotkey.key ||
          JSON.stringify(h.modifiers?.sort()) !==
            JSON.stringify(hotkey.modifiers?.sort())
      );
    };
  }, []);

  const unregisterHotkey = React.useCallback(
    (key: string, modifiers?: ModifierKey[]) => {
      hotkeysRef.current = hotkeysRef.current.filter(
        (h) =>
          h.key !== key ||
          JSON.stringify(h.modifiers?.sort()) !== JSON.stringify(modifiers?.sort())
      );
    },
    []
  );

  const getHotkeys = React.useCallback(() => {
    return [...hotkeysRef.current];
  }, []);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isModalOpen) return;

      const activeElement = document.activeElement;
      const isInput =
        activeElement instanceof HTMLInputElement ||
        activeElement instanceof HTMLTextAreaElement ||
        activeElement?.getAttribute("contenteditable") === "true";

      for (const hotkey of hotkeysRef.current) {
        const modifiers = hotkey.modifiers || [];
        const hasCtrl = modifiers.includes("ctrl");
        const hasAlt = modifiers.includes("alt");
        const hasShift = modifiers.includes("shift");
        const hasMeta = modifiers.includes("meta");

        const matchCtrl = hasCtrl ? e.ctrlKey || e.metaKey : !e.ctrlKey && !e.metaKey;
        const matchAlt = hasAlt ? e.altKey : !e.altKey;
        const matchShift = hasShift ? e.shiftKey : !e.shiftKey;
        const matchMeta = hasMeta ? e.metaKey : true;

        const keyMatch =
          e.key.toLowerCase() === hotkey.key.toLowerCase() ||
          e.code.toLowerCase() === hotkey.key.toLowerCase();

        if (keyMatch && matchCtrl && matchAlt && matchShift && matchMeta) {
          if (isInput && !hasCtrl && !hasAlt && !hasMeta) {
            continue;
          }

          e.preventDefault();
          hotkey.action();
          return;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isModalOpen]);

  return (
    <HotkeysContext.Provider
      value={{
        registerHotkey,
        unregisterHotkey,
        getHotkeys,
        isModalOpen,
        setModalOpen,
      }}
    >
      {children}
    </HotkeysContext.Provider>
  );
}

interface UseHotkeyOptions {
  key: string;
  modifiers?: ModifierKey[];
  action: () => void;
  description?: string;
  group?: string;
  enabled?: boolean;
}

function useHotkey({
  key,
  modifiers,
  action,
  description = "",
  group,
  enabled = true,
}: UseHotkeyOptions) {
  const { registerHotkey, unregisterHotkey } = useHotkeys();

  React.useEffect(() => {
    if (!enabled) return;

    const hotkey: Hotkey = {
      key,
      modifiers,
      description,
      action,
      group,
    };

    const unregister = registerHotkey(hotkey);
    return () => unregister();
  }, [key, modifiers, action, description, group, enabled, registerHotkey]);
}

function formatHotkey(hotkey: Hotkey): string {
  const parts: string[] = [];

  const isMac =
    typeof navigator !== "undefined" && navigator.platform.toUpperCase().indexOf("MAC") >= 0;

  if (hotkey.modifiers) {
    for (const mod of hotkey.modifiers) {
      switch (mod) {
        case "ctrl":
          parts.push(isMac ? "⌘" : "Ctrl");
          break;
        case "alt":
          parts.push(isMac ? "⌥" : "Alt");
          break;
        case "shift":
          parts.push("⇧");
          break;
        case "meta":
          parts.push(isMac ? "⌘" : "Win");
          break;
      }
    }
  }

  parts.push(hotkey.key.toUpperCase());

  return parts.join(isMac ? "" : "+");
}

function HotkeyBadge({
  hotkey,
  className,
}: {
  hotkey: Hotkey;
  className?: string;
}) {
  return (
    <kbd
      className={cn(
        "inline-flex items-center justify-center px-1.5 py-0.5 text-xs font-mono bg-muted border rounded",
        className
      )}
    >
      {formatHotkey(hotkey)}
    </kbd>
  );
}

function HotkeyHelpModal() {
  const { getHotkeys, isModalOpen, setModalOpen } = useHotkeys();

  useHotkey({
    key: "?",
    modifiers: ["shift"],
    action: () => setModalOpen(!isModalOpen),
    description: "Toggle keyboard shortcuts help",
    group: "General",
  });

  if (!isModalOpen) return null;

  const hotkeys = getHotkeys();
  const groupedHotkeys = hotkeys.reduce(
    (acc, hotkey) => {
      const group = hotkey.group || "General";
      if (!acc[group]) acc[group] = [];
      acc[group].push(hotkey);
      return acc;
    },
    {} as Record<string, Hotkey[]>
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={() => setModalOpen(false)}
    >
      <div
        className="bg-background border rounded-lg shadow-lg max-w-lg w-full mx-4 max-h-[80vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
          <button
            onClick={() => setModalOpen(false)}
            className="text-muted-foreground hover:text-foreground"
          >
            ✕
          </button>
        </div>
        <div className="p-4">
          {Object.entries(groupedHotkeys).map(([group, keys]) => (
            <div key={group} className="mb-4 last:mb-0">
              <h3 className="text-sm font-medium text-muted-foreground mb-2">
                {group}
              </h3>
              <div className="space-y-2">
                {keys.map((hotkey, index) => (
                  <div
                    key={`${hotkey.key}-${index}`}
                    className="flex items-center justify-between"
                  >
                    <span className="text-sm">{hotkey.description}</span>
                    <HotkeyBadge hotkey={hotkey} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface HotkeyMenuItemProps {
  children: React.ReactNode;
  hotkey: Hotkey;
  className?: string;
}

function HotkeyMenuItem({ children, hotkey, className }: HotkeyMenuItemProps) {
  return (
    <div className={cn("flex items-center justify-between", className)}>
      <span>{children}</span>
      <HotkeyBadge hotkey={hotkey} />
    </div>
  );
}

export {
  HotkeysProvider,
  useHotkeys,
  useHotkey,
  HotkeyHelpModal,
  HotkeyBadge,
  HotkeyMenuItem,
  formatHotkey,
};

export type { Hotkey, ModifierKey, HotkeyConfig };
