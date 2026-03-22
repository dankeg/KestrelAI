import React, { useEffect } from "react";
import { Brain, Globe, X, Zap } from "lucide-react";

import { type AppSettingsFormState, DEFAULT_SETTINGS } from "./settings";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  settings: AppSettingsFormState;
  onChange: (patch: Partial<AppSettingsFormState>) => void;
  onSave: () => void;
  linkModels: boolean;
  onLinkModelsChange: (next: boolean) => void;
  availableModels: string[];
  modelsLoading: boolean;
  modelsError: string | null;
  onRefreshModels: () => void;
}

export default function SettingsModal({
  open,
  onClose,
  settings,
  onChange,
  onSave,
  linkModels,
  onLinkModelsChange,
  availableModels,
  modelsLoading,
  modelsError,
  onRefreshModels,
}: SettingsModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleExecutionModelChange = (value: string) => {
    if (linkModels) {
      onChange({
        executionModelName: value,
        controlModelName: value,
        modelName: value,
      });
      return;
    }
    onChange({ executionModelName: value });
  };

  const handleControlModelChange = (value: string) => {
    onChange({
      controlModelName: value,
      modelName: value,
    });
  };

  const SelFrame = ({
    selected,
    children,
  }: {
    selected: boolean;
    children: React.ReactNode;
  }) => (
    <div
      className={[
        "p-4 rounded-xl border-2 transition-all bg-white",
        selected
          ? "theme-border-primary-500 shadow-md ring-2 ring-theme-border-primary-200"
          : "border-gray-200 hover:theme-border-primary-300",
      ].join(" ")}
    >
      {children}
    </div>
  );

  return (
    <div className="fixed inset-0 z-[100]">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-4xl bg-white rounded-2xl shadow-2xl border theme-border-primary-200 overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 theme-bg-primary-50 border-b theme-border-primary-200">
            <h2 className="text-lg font-bold text-gray-900">Settings</h2>
            <button onClick={onClose} className="p-2 rounded hover:theme-bg-primary-100">
              <X className="w-5 h-5 theme-text-primary-700" />
            </button>
          </div>

          <div className="p-6 space-y-6">
            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Model Provider</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <button onClick={() => onChange({ llmProvider: "ollama" })}>
                  <SelFrame selected={settings.llmProvider === "ollama"}>
                    <div className="font-semibold text-gray-900">Ollama</div>
                    <div className="text-xs text-gray-600">
                      Discover models from the configured local or Docker Ollama runtime.
                    </div>
                  </SelFrame>
                </button>

                <button onClick={() => onChange({ llmProvider: "openai_compatible" })}>
                  <SelFrame selected={settings.llmProvider === "openai_compatible"}>
                    <div className="font-semibold text-gray-900">OpenAI-Compatible API</div>
                    <div className="text-xs text-gray-600">
                      Connect to any `/v1/chat/completions` and `/v1/models` compatible endpoint.
                    </div>
                  </SelFrame>
                </button>
              </div>
            </section>

            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Theme</h3>
              <div className="grid grid-cols-2 gap-3">
                <button onClick={() => onChange({ theme: "amber" })}>
                  <SelFrame selected={settings.theme === "amber"}>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-4 h-4 rounded-full bg-gradient-to-r from-amber-400 to-orange-500"></div>
                      <div className="font-semibold text-gray-900">Amber</div>
                    </div>
                    <div className="text-xs text-gray-600">
                      Warm orange and amber tones for a cozy feel.
                    </div>
                  </SelFrame>
                </button>

                <button onClick={() => onChange({ theme: "blue" })}>
                  <SelFrame selected={settings.theme === "blue"}>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-4 h-4 rounded-full bg-gradient-to-r from-blue-400 to-cyan-500"></div>
                      <div className="font-semibold text-gray-900">Blue</div>
                    </div>
                    <div className="text-xs text-gray-600">
                      Cool blue and cyan tones for a professional look.
                    </div>
                  </SelFrame>
                </button>
              </div>
            </section>

            {settings.llmProvider === "ollama" ? (
              <section>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Ollama Runtime</h3>
                <div className="grid grid-cols-2 gap-3">
                  <button onClick={() => onChange({ ollamaMode: "local" })}>
                    <SelFrame selected={settings.ollamaMode === "local"}>
                      <div className="font-semibold text-gray-900">Local</div>
                      <div className="text-xs text-gray-600">
                        Use the Ollama instance running on this machine.
                      </div>
                    </SelFrame>
                  </button>

                  <button onClick={() => onChange({ ollamaMode: "docker" })}>
                    <SelFrame selected={settings.ollamaMode === "docker"}>
                      <div className="font-semibold text-gray-900">Docker</div>
                      <div className="text-xs text-gray-600">
                        Route requests to the Dockerized Ollama service.
                      </div>
                    </SelFrame>
                  </button>
                </div>
              </section>
            ) : (
              <section>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">OpenAI-Compatible API</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="p-4 rounded-xl border-2 border-gray-200 bg-white md:col-span-2">
                    <label className="block text-xs font-semibold text-gray-600 mb-2 uppercase">
                      Base URL
                    </label>
                    <input
                      type="text"
                      value={settings.openaiBaseUrl}
                      onChange={(e) => onChange({ openaiBaseUrl: e.target.value })}
                      placeholder="https://api.openai.com/v1"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm"
                    />
                    <p className="mt-2 text-xs text-gray-500">
                      Example: `https://api.openai.com/v1`, LM Studio, vLLM, or another OpenAI-format endpoint.
                    </p>
                  </div>

                  <div className="p-4 rounded-xl border-2 border-gray-200 bg-white md:col-span-2">
                    <div className="flex items-center justify-between mb-2 gap-3">
                      <label className="block text-xs font-semibold text-gray-600 uppercase">
                        API Key
                      </label>
                      {settings.openaiApiKeySet ? (
                        <button
                          type="button"
                          onClick={() =>
                            onChange({
                              openaiApiKey: "",
                              openaiApiKeySet: false,
                              clearOpenaiApiKey: true,
                            })
                          }
                          className="px-2 py-1 text-xs rounded border border-gray-300 hover:bg-gray-50"
                        >
                          Clear saved key
                        </button>
                      ) : null}
                    </div>
                    <input
                      type="password"
                      value={settings.openaiApiKey}
                      onChange={(e) =>
                        onChange({
                          openaiApiKey: e.target.value,
                          openaiApiKeySet:
                            settings.openaiApiKeySet ||
                            e.target.value.trim().length > 0,
                          clearOpenaiApiKey: false,
                        })
                      }
                      placeholder={
                        settings.openaiApiKeySet
                          ? "Saved key is configured. Enter a new key to replace it."
                          : "Optional for local runtimes"
                      }
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm"
                    />
                    <p className="mt-2 text-xs text-gray-500">
                      Stored server-side only. Local browser persistence keeps only whether a key is configured.
                    </p>
                  </div>
                </div>
              </section>
            )}

            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Orchestrator</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button onClick={() => onChange({ orchestrator: "hummingbird" })}>
                  <SelFrame selected={settings.orchestrator === "hummingbird"}>
                    <div className="flex items-center gap-2 mb-1">
                      <Zap className="w-4 h-4 text-amber-600" />
                      <div className="font-semibold text-gray-900">Hummingbird</div>
                    </div>
                    <div className="text-xs text-gray-600">
                      Fast, focused answers on a single prompt with minimal exploration.
                    </div>
                  </SelFrame>
                </button>

                <button onClick={() => onChange({ orchestrator: "kestrel" })}>
                  <SelFrame selected={settings.orchestrator === "kestrel"}>
                    <div className="flex items-center gap-2 mb-1">
                      <Brain className="w-4 h-4 text-amber-600" />
                      <div className="font-semibold text-gray-900">Kestrel</div>
                    </div>
                    <div className="text-xs text-gray-600">
                      Balanced exploration: expands key angles and synthesizes medium-depth insights.
                    </div>
                  </SelFrame>
                </button>

                <button onClick={() => onChange({ orchestrator: "albatross" })}>
                  <SelFrame selected={settings.orchestrator === "albatross"}>
                    <div className="flex items-center gap-2 mb-1">
                      <Globe className="w-4 h-4 text-amber-600" />
                      <div className="font-semibold text-gray-900">Albatross</div>
                    </div>
                    <div className="text-xs text-gray-600">
                      Long-horizon research: new leads, deep dives, and cross-topic synthesis.
                    </div>
                  </SelFrame>
                </button>
              </div>
            </section>

            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Model & Context</h3>
              <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50/70 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-sm font-semibold text-gray-900">Model role split</div>
                    <div className="text-xs text-gray-600 mt-1">
                      Execution handles high-volume search and extraction. Control handles orchestration, replanning, and the final report.
                    </div>
                  </div>
                  <label className="inline-flex items-center gap-3 text-sm font-medium text-gray-800">
                    <input
                      type="checkbox"
                      checked={linkModels}
                      onChange={(e) => {
                        const nextLinked = e.target.checked;
                        onLinkModelsChange(nextLinked);
                        if (nextLinked) {
                          onChange({
                            controlModelName: settings.executionModelName,
                            modelName: settings.executionModelName,
                          });
                        }
                      }}
                      className="h-4 w-4 rounded border-gray-300 text-amber-600 focus:ring-amber-500"
                    />
                    Use one model for both roles
                  </label>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="p-4 rounded-xl border-2 border-gray-200 bg-white">
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-xs font-semibold text-gray-600 uppercase">
                      Execution Model
                    </label>
                    <button
                      type="button"
                      onClick={onRefreshModels}
                      className="px-2 py-1 text-xs rounded border border-gray-300 hover:bg-gray-50"
                    >
                      Refresh
                    </button>
                  </div>
                  {availableModels.length > 0 ? (
                    <select
                      value={settings.executionModelName}
                      onChange={(e) => handleExecutionModelChange(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm bg-white"
                    >
                      {availableModels.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={settings.executionModelName}
                      onChange={(e) => handleExecutionModelChange(e.target.value)}
                      placeholder="gemma3:12b"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm"
                    />
                  )}
                  <p className="mt-2 text-xs text-gray-500">
                    {modelsLoading
                      ? settings.llmProvider === "ollama"
                        ? "Loading models from Ollama..."
                        : "Loading models from OpenAI-compatible API..."
                      : modelsError
                        ? `Model discovery error: ${modelsError}`
                        : "Used for worker search, fetching, and extraction loops."}
                  </p>
                </div>

                <div className="p-4 rounded-xl border-2 border-gray-200 bg-white">
                  <label className="block text-xs font-semibold text-gray-600 mb-2 uppercase">
                    Control / Report Model
                  </label>
                  {availableModels.length > 0 ? (
                    <select
                      value={
                        linkModels
                          ? settings.executionModelName
                          : settings.controlModelName
                      }
                      onChange={(e) => handleControlModelChange(e.target.value)}
                      disabled={linkModels}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm bg-white disabled:bg-gray-100 disabled:text-gray-500"
                    >
                      {availableModels.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={
                        linkModels
                          ? settings.executionModelName
                          : settings.controlModelName
                      }
                      onChange={(e) => handleControlModelChange(e.target.value)}
                      disabled={linkModels}
                      placeholder="gemma3:12b"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm disabled:bg-gray-100 disabled:text-gray-500"
                    />
                  )}
                  <p className="mt-2 text-xs text-gray-500">
                    {linkModels
                      ? "Currently linked to the execution model."
                      : "Used for orchestration, replanning, and final report synthesis. If you want one stronger model, spend it here."}
                  </p>
                </div>

                <div className="p-4 rounded-xl border-2 border-gray-200 bg-white">
                  <label className="block text-xs font-semibold text-gray-600 mb-2 uppercase">
                    Max Context Tokens
                  </label>
                  <input
                    type="number"
                    min={2048}
                    max={262144}
                    step={1024}
                    value={settings.maxContextTokens}
                    onChange={(e) =>
                      onChange({
                        maxContextTokens: Number(
                          e.target.value || DEFAULT_SETTINGS.maxContextTokens
                        ),
                      })
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-theme-border-primary-500 focus:border-transparent text-sm"
                  />
                  <p className="mt-2 text-xs text-gray-500">
                    Controls token budget for retrieval/planning context windows.
                  </p>
                </div>
              </div>
            </section>
          </div>

          <div className="px-6 py-4 bg-gradient-to-r from-amber-50 to-orange-50 border-t border-amber-200 flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-amber-300 text-amber-900 font-semibold hover:bg-amber-100 transition"
            >
              Cancel
            </button>
            <button
              onClick={onSave}
              className="px-4 py-2 rounded-lg bg-amber-600 text-white font-semibold hover:bg-amber-700 transition"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
