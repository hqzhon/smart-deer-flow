// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

export function deepClone<T>(value: T): T {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (error) {
    console.error('Failed to deep clone object:', value, error);
    // Return the original value as fallback
    return value;
  }
}
