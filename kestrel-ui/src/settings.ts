export type OllamaMode = "local" | "docker";
export type LlmProvider = "ollama" | "openai_compatible";
export type Orchestrator = "hummingbird" | "kestrel" | "albatross";
export type Theme = "amber" | "blue";

export interface AppSettings {
  llmProvider: LlmProvider;
  ollamaMode: OllamaMode;
  orchestrator: Orchestrator;
  theme: Theme;
  modelName: string;
  executionModelName: string;
  controlModelName: string;
  openaiBaseUrl: string;
  openaiApiKeySet: boolean;
  maxContextTokens: number;
}

export interface AppSettingsFormState extends AppSettings {
  openaiApiKey: string;
  clearOpenaiApiKey: boolean;
}

export interface AppSettingsSavePayload {
  llmProvider: LlmProvider;
  ollamaMode: OllamaMode;
  orchestrator: Orchestrator;
  theme: Theme;
  modelName: string;
  executionModelName: string;
  controlModelName: string;
  openaiBaseUrl: string;
  openaiApiKey: string;
  clearOpenaiApiKey: boolean;
  maxContextTokens: number;
}

export interface AvailableModelsResponse {
  provider: LlmProvider;
  mode?: OllamaMode | null;
  baseUrl: string;
  models: string[];
  selected?: string | null;
  error?: string | null;
}

export const DEFAULT_SETTINGS: AppSettingsFormState = {
  llmProvider: "ollama",
  ollamaMode: "local",
  orchestrator: "kestrel",
  theme: "amber",
  modelName: "gemma3:12b",
  executionModelName: "gemma3:12b",
  controlModelName: "gemma3:12b",
  openaiBaseUrl: "https://api.openai.com/v1",
  openaiApiKey: "",
  openaiApiKeySet: false,
  clearOpenaiApiKey: false,
  maxContextTokens: 32768,
};

export const toSettingsFormState = (
  settings: Partial<AppSettings> | Partial<AppSettingsFormState>
): AppSettingsFormState => {
  const base = {
    ...DEFAULT_SETTINGS,
    ...settings,
  };
  const legacyModelName = (settings.modelName || base.modelName || DEFAULT_SETTINGS.modelName).trim();
  return {
    ...base,
    modelName: legacyModelName,
    executionModelName: (settings.executionModelName || legacyModelName).trim() || legacyModelName,
    controlModelName: (settings.controlModelName || legacyModelName).trim() || legacyModelName,
    openaiApiKey: "",
    clearOpenaiApiKey: false,
  };
};

export const sanitizeSettingsForStorage = (
  settings: AppSettingsFormState
): AppSettings => ({
  llmProvider: settings.llmProvider,
  ollamaMode: settings.ollamaMode,
  orchestrator: settings.orchestrator,
  theme: settings.theme,
  modelName: settings.modelName,
  executionModelName: settings.executionModelName,
  controlModelName: settings.controlModelName,
  openaiBaseUrl: settings.openaiBaseUrl,
  openaiApiKeySet: settings.openaiApiKeySet,
  maxContextTokens: settings.maxContextTokens,
});

export const buildSettingsSavePayload = (
  settings: AppSettingsFormState
): AppSettingsSavePayload => ({
  llmProvider: settings.llmProvider,
  ollamaMode: settings.ollamaMode,
  orchestrator: settings.orchestrator,
  theme: settings.theme,
  modelName:
    settings.controlModelName.trim() ||
    settings.executionModelName.trim() ||
    settings.modelName.trim(),
  executionModelName: settings.executionModelName.trim(),
  controlModelName: settings.controlModelName.trim(),
  openaiBaseUrl: settings.openaiBaseUrl,
  openaiApiKey: settings.openaiApiKey.trim(),
  clearOpenaiApiKey: settings.clearOpenaiApiKey,
  maxContextTokens: settings.maxContextTokens,
});

export const normalizeSettingsFormState = (
  settings: AppSettingsFormState
): AppSettingsFormState => {
  const next = { ...settings };
  const normalizedLegacyModelName =
    (next.modelName || DEFAULT_SETTINGS.modelName).trim() ||
    DEFAULT_SETTINGS.modelName;
  next.executionModelName =
    (next.executionModelName || normalizedLegacyModelName).trim() ||
    normalizedLegacyModelName;
  next.controlModelName =
    (next.controlModelName || normalizedLegacyModelName).trim() ||
    normalizedLegacyModelName;
  next.modelName = next.controlModelName || next.executionModelName || normalizedLegacyModelName;
  const parsedMaxContext = Number(next.maxContextTokens);
  next.maxContextTokens = Number.isFinite(parsedMaxContext)
    ? Math.max(2048, Math.min(262144, Math.round(parsedMaxContext)))
    : DEFAULT_SETTINGS.maxContextTokens;
  return next;
};
