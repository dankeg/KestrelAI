import { describe, expect, it } from "vitest";

import {
  DEFAULT_SETTINGS,
  buildSettingsSavePayload,
  normalizeSettingsFormState,
  sanitizeSettingsForStorage,
  toSettingsFormState,
} from "./settings";

describe("settings helpers", () => {
  it("drops transient secret fields when building form state", () => {
    const state = toSettingsFormState({
      ...DEFAULT_SETTINGS,
      openaiApiKey: "sk-secret",
      clearOpenaiApiKey: true,
    });

    expect(state.openaiApiKey).toBe("");
    expect(state.clearOpenaiApiKey).toBe(false);
  });

  it("builds save payload with trimmed secret input", () => {
    const payload = buildSettingsSavePayload({
      ...DEFAULT_SETTINGS,
      llmProvider: "openai_compatible",
      executionModelName: "  gpt-4.1-mini  ",
      controlModelName: "  o4-mini  ",
      openaiApiKey: "  sk-secret  ",
      clearOpenaiApiKey: true,
    });

    expect(payload.modelName).toBe("o4-mini");
    expect(payload.executionModelName).toBe("gpt-4.1-mini");
    expect(payload.controlModelName).toBe("o4-mini");
    expect(payload.openaiApiKey).toBe("sk-secret");
    expect(payload.clearOpenaiApiKey).toBe(true);
  });

  it("sanitizes settings for browser storage", () => {
    const stored = sanitizeSettingsForStorage({
      ...DEFAULT_SETTINGS,
      openaiApiKey: "sk-secret",
      openaiApiKeySet: true,
      clearOpenaiApiKey: true,
    });

    expect("openaiApiKey" in stored).toBe(false);
    expect("clearOpenaiApiKey" in stored).toBe(false);
    expect(stored.openaiApiKeySet).toBe(true);
  });

  it("normalizes model name and context bounds", () => {
    const normalized = normalizeSettingsFormState({
      ...DEFAULT_SETTINGS,
      modelName: "   ",
      executionModelName: "   ",
      controlModelName: "  o4-mini  ",
      maxContextTokens: 999999,
    });

    expect(normalized.executionModelName).toBe(DEFAULT_SETTINGS.modelName);
    expect(normalized.controlModelName).toBe("o4-mini");
    expect(normalized.modelName).toBe("o4-mini");
    expect(normalized.maxContextTokens).toBe(262144);
  });

  it("inherits role-specific models from the legacy model when absent", () => {
    const normalized = toSettingsFormState({
      ...DEFAULT_SETTINGS,
      modelName: "qwen3:14b",
      executionModelName: "",
      controlModelName: "",
    });

    expect(normalized.executionModelName).toBe("qwen3:14b");
    expect(normalized.controlModelName).toBe("qwen3:14b");
  });
});
