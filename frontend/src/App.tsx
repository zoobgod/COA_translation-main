import { FormEvent, Fragment, useEffect, useMemo, useState } from "react";

type Capabilities = {
  has_ocr: boolean;
  has_camelot?: boolean;
  has_tabula?: boolean;
};

type TemplateHints = {
  placeholders?: string[];
  headings?: string[];
};

type ExtractionResult = {
  success: boolean;
  method?: string;
  page_count?: number;
  text: string;
  table_supplement?: string;
  error?: string;
  template_hints?: TemplateHints | null;
};

type TranslationResult = {
  success: boolean;
  translated_text: string;
  sections?: Record<string, unknown>;
  template_fields?: Record<string, unknown>;
  template_heading_map?: Record<string, unknown>;
  model_used?: string;
  chunks_translated?: number;
  error?: string;
};

type DiffRow = {
  left: string;
  right: string;
  kind: "eq" | "chg" | "add" | "del";
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

function buildUrl(path: string): string {
  return `${API_BASE}${path}`;
}

function readFilenameFromContentDisposition(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const match = value.match(/filename="?([^";]+)"?/i);
  return match ? match[1] : null;
}

async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function buildDiffRows(source: string, translated: string, maxLines = 250): DiffRow[] {
  const src = source.split(/\r?\n/).slice(0, maxLines);
  const dst = translated.split(/\r?\n/).slice(0, maxLines);
  const total = Math.max(src.length, dst.length);

  const rows: DiffRow[] = [];
  for (let i = 0; i < total; i += 1) {
    const left = src[i] ?? "";
    const right = dst[i] ?? "";

    if (left && right) {
      rows.push({ left, right, kind: left === right ? "eq" : "chg" });
    } else if (left && !right) {
      rows.push({ left, right: "", kind: "del" });
    } else {
      rows.push({ left: "", right, kind: "add" });
    }
  }
  return rows;
}

function StepHeader({ step, title, subtitle }: { step: string; title: string; subtitle: string }) {
  return (
    <div className="mb-4">
      <span className="step-pill">{step}</span>
      <h2 className="mt-2 text-xl font-semibold tracking-tight text-fg md:text-2xl">{title}</h2>
      <p className="mt-1 text-sm text-fgMuted">{subtitle}</p>
    </div>
  );
}

export default function App() {
  const [apiKey, setApiKey] = useState("");
  const [modelChoice, setModelChoice] = useState("gpt-4.1");
  const [customModel, setCustomModel] = useState("");

  const [coaFile, setCoaFile] = useState<File | null>(null);
  const [templateFile, setTemplateFile] = useState<File | null>(null);

  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);

  const [extracting, setExtracting] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [generating, setGenerating] = useState(false);

  const [extraction, setExtraction] = useState<ExtractionResult | null>(null);
  const [translation, setTranslation] = useState<TranslationResult | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const selectedModel = useMemo(() => {
    if (modelChoice === "custom") {
      return customModel.trim();
    }
    return modelChoice;
  }, [modelChoice, customModel]);

  const diffRows = useMemo(() => {
    if (!extraction?.text || !translation?.translated_text) {
      return [];
    }
    return buildDiffRows(extraction.text, translation.translated_text);
  }, [extraction?.text, translation?.translated_text]);

  useEffect(() => {
    const controller = new AbortController();

    fetch(buildUrl("/api/capabilities"), { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: Capabilities | null) => {
        if (data) {
          setCapabilities(data);
        }
      })
      .catch(() => undefined);

    return () => controller.abort();
  }, []);

  const onExtract = async (event: FormEvent) => {
    event.preventDefault();
    setErrorMessage(null);
    if (!coaFile) {
      setErrorMessage("Upload a COA file first.");
      return;
    }

    setExtracting(true);
    setTranslation(null);

    try {
      const formData = new FormData();
      formData.append("file", coaFile);
      if (templateFile) {
        formData.append("template", templateFile);
      }
      if (apiKey.trim()) {
        formData.append("api_key", apiKey.trim());
        formData.append("vision_ocr_model", "gpt-4o-mini");
      }

      const response = await fetch(buildUrl("/api/extract"), {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as ExtractionResult;

      setExtraction(data);
      if (!data.success) {
        setErrorMessage(data.error ?? "Text extraction failed.");
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Extraction request failed.");
    } finally {
      setExtracting(false);
    }
  };

  const onTranslate = async () => {
    setErrorMessage(null);

    if (!extraction?.success || !extraction.text) {
      setErrorMessage("Run extraction before translation.");
      return;
    }
    if (!apiKey.trim()) {
      setErrorMessage("Enter your OpenAI API key.");
      return;
    }
    if (!selectedModel) {
      setErrorMessage("Set a valid model ID.");
      return;
    }

    setTranslating(true);

    try {
      const response = await fetch(buildUrl("/api/translate"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: extraction.text,
          api_key: apiKey.trim(),
          model: selectedModel,
          template_hints: extraction.template_hints ?? null,
          table_supplement: extraction.table_supplement ?? "",
        }),
      });

      const data = (await response.json()) as TranslationResult;
      setTranslation(data);
      if (!data.success) {
        setErrorMessage(data.error ?? "Translation failed.");
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Translation request failed.");
    } finally {
      setTranslating(false);
    }
  };

  const onGenerateDoc = async () => {
    setErrorMessage(null);

    if (!translation?.success || !translation.sections || !extraction || !coaFile) {
      setErrorMessage("Complete extraction and translation first.");
      return;
    }

    setGenerating(true);

    try {
      const templateBase64 = templateFile ? await fileToBase64(templateFile) : null;

      const response = await fetch(buildUrl("/api/generate-doc"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sections: translation.sections,
          original_filename: coaFile.name,
          extraction_method: extraction.method ?? "unknown",
          model_used: translation.model_used ?? selectedModel,
          template_fields: translation.template_fields ?? {},
          template_heading_map: translation.template_heading_map ?? {},
          user_template_base64: templateBase64,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Document generation failed.");
      }

      const blob = await response.blob();
      const contentDisposition = response.headers.get("content-disposition");
      const suggestedName =
        readFilenameFromContentDisposition(contentDisposition) ??
        `${coaFile.name.replace(/\.[^.]+$/, "")}_RU.docx`;

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = suggestedName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Download failed.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <>
      <div className="ambient-layer">
        <div className="ambient-blob left-[-260px] top-[120px] h-[800px] w-[620px] animate-float bg-[radial-gradient(circle,rgba(142,95,239,0.42)_0%,rgba(142,95,239,0)_70%)]" />
        <div className="ambient-blob right-[-160px] top-[180px] h-[760px] w-[560px] animate-float bg-[radial-gradient(circle,rgba(94,106,210,0.55)_0%,rgba(94,106,210,0)_72%)] [animation-duration:12s]" />
        <div className="ambient-blob left-1/2 top-[-420px] h-[1200px] w-[920px] -translate-x-1/2 animate-pulseGlow bg-[radial-gradient(circle,rgba(94,106,210,0.45)_0%,rgba(94,106,210,0)_67%)]" />
      </div>

      <main className="mx-auto max-w-[1220px] px-4 py-6 md:px-8 md:py-10">
        <header className="card p-6 md:p-8">
          <span className="inline-flex rounded-full border border-[#5E6AD2]/40 bg-[#5E6AD2]/15 px-3 py-1 text-[11px] uppercase tracking-[0.14em] text-[#c8ceff]">
            Pharmacopeia Workflow
          </span>
          <h1 className="mt-4 text-4xl font-semibold leading-tight tracking-[-0.03em] text-transparent md:text-6xl bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text">
            Pharmaceutical <span className="bg-[linear-gradient(90deg,#5E6AD2_0%,#8f98e8_46%,#5E6AD2_100%)] bg-[length:200%] bg-clip-text text-transparent animate-shimmer">COA Translator</span>
          </h1>
          <p className="mt-3 max-w-3xl text-base leading-relaxed text-fgMuted">
            Rebuilt UI with a proper API architecture. Core extraction, translation, and DOCX logic remain unchanged.
          </p>
        </header>

        <div className="mt-6 grid gap-6 lg:grid-cols-[320px_1fr]">
          <aside className="space-y-4 lg:sticky lg:top-4 lg:h-fit">
            <section className="card p-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-fgMuted">Settings</h2>
              <div className="mt-3 space-y-3">
                <div>
                  <label className="mb-1 block text-xs text-fgMuted">OpenAI API Key</label>
                  <input
                    className="input"
                    type="password"
                    placeholder="sk-..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs text-fgMuted">Model</label>
                  <select
                    className="select"
                    value={modelChoice}
                    onChange={(e) => setModelChoice(e.target.value)}
                  >
                    <option value="gpt-4.1">gpt-4.1</option>
                    <option value="gpt-4o">gpt-4o</option>
                    <option value="gpt-4o-mini">gpt-4o-mini</option>
                    <option value="custom">Custom model ID</option>
                  </select>
                </div>

                {modelChoice === "custom" ? (
                  <div>
                    <label className="mb-1 block text-xs text-fgMuted">Custom model ID</label>
                    <input
                      className="input"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                      placeholder="gpt-5 or other model id"
                    />
                  </div>
                ) : null}
              </div>
            </section>

            <section className="card p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-fgMuted">Runtime</h3>
              <div className="mt-3 space-y-2 text-sm text-fgMuted">
                <p>
                  OCR: <span className="text-fg">{capabilities?.has_ocr ? "available" : "missing"}</span>
                </p>
                {!capabilities?.has_ocr ? (
                  <p className="text-xs">
                    AI OCR fallback is used automatically during extraction when API key is provided.
                  </p>
                ) : null}
                <p>
                  Tables: <span className="text-fg">{capabilities?.has_camelot || capabilities?.has_tabula ? "advanced extractors ready" : "baseline only"}</span>
                </p>
              </div>
            </section>
          </aside>

          <section className="space-y-4">
            <form className="card p-5 md:p-6" onSubmit={onExtract}>
              <StepHeader
                step="Step 1"
                title="Upload COA + Optional Template"
                subtitle="Supports PDF and image files. Template stays optional and is applied at DOCX generation."
              />

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-1 block text-xs text-fgMuted">COA file</label>
                  <input
                    className="file"
                    type="file"
                    accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
                    onChange={(e) => setCoaFile(e.target.files?.[0] ?? null)}
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs text-fgMuted">Word structure template (.docx, optional)</label>
                  <input
                    className="file"
                    type="file"
                    accept=".docx"
                    onChange={(e) => setTemplateFile(e.target.files?.[0] ?? null)}
                  />
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button className="btn-primary min-w-[180px]" disabled={extracting || !coaFile} type="submit">
                  {extracting ? "Extracting..." : "Extract Text"}
                </button>
                <span className="text-sm text-fgMuted">
                  {coaFile ? `${coaFile.name} (${(coaFile.size / (1024 * 1024)).toFixed(2)} MB)` : "No file selected"}
                </span>
              </div>
            </form>

            {extraction ? (
              <section className="card p-5 md:p-6">
                <StepHeader
                  step="Step 2"
                  title="Extraction Preview"
                  subtitle="Validate full source text before translation."
                />

                {extraction.success ? (
                  <>
                    <p className="mb-3 text-sm text-fgMuted">
                      Method: <span className="text-fg">{extraction.method}</span> | Pages: <span className="text-fg">{extraction.page_count ?? 0}</span> | Characters: <span className="text-fg">{extraction.text.length.toLocaleString()}</span>
                    </p>
                    <textarea className="textarea h-64" readOnly value={extraction.text} />

                    {extraction.template_hints ? (
                      <p className="mt-3 text-xs text-fgMuted">
                        Template hints loaded: {(extraction.template_hints.placeholders ?? []).length} placeholders, {(extraction.template_hints.headings ?? []).length} heading hints.
                      </p>
                    ) : null}
                  </>
                ) : (
                  <p className="rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                    {extraction.error ?? "Failed to extract text."}
                  </p>
                )}
              </section>
            ) : null}

            {extraction?.success ? (
              <section className="card p-5 md:p-6">
                <StepHeader
                  step="Step 3"
                  title="Translate to Russian"
                  subtitle="High-fidelity full translation with glossary and section mapping."
                />

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    className="btn-primary min-w-[210px]"
                    type="button"
                    onClick={onTranslate}
                    disabled={translating || !apiKey || !selectedModel}
                  >
                    {translating ? "Translating..." : "Translate to Russian"}
                  </button>
                  <span className="text-sm text-fgMuted">Model: {selectedModel || "set custom model"}</span>
                </div>
              </section>
            ) : null}

            {translation ? (
              <section className="card p-5 md:p-6">
                <StepHeader
                  step="Step 3.5"
                  title="Bilingual Review"
                  subtitle="Side-by-side verification before exporting DOCX."
                />

                {translation.success ? (
                  <>
                    <p className="mb-3 text-sm text-fgMuted">
                      Translation complete using <span className="text-fg">{translation.model_used ?? selectedModel}</span>
                    </p>

                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <label className="mb-1 block text-xs text-fgMuted">English source</label>
                        <textarea className="textarea h-64" readOnly value={extraction?.text ?? ""} />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-fgMuted">Russian translation</label>
                        <textarea className="textarea h-64" readOnly value={translation.translated_text} />
                      </div>
                    </div>

                    <div className="mt-4">
                      <label className="mb-2 block text-xs text-fgMuted">Line-by-line diff (first 250 lines)</label>
                      <div className="max-h-[420px] overflow-auto rounded-xl border border-white/10 bg-black/20 p-3">
                        <div className="diff-grid">
                          {diffRows.map((row, idx) => (
                            <Fragment key={`row-${idx}`}>
                              <div
                                className={`diff-row ${
                                  row.kind === "eq"
                                    ? "diff-eq"
                                    : row.kind === "chg"
                                      ? "diff-chg"
                                      : row.kind === "del"
                                        ? "diff-del"
                                        : "diff-add"
                                }`}
                              >
                                {row.left || "-"}
                              </div>
                              <div
                                className={`diff-row ${
                                  row.kind === "eq"
                                    ? "diff-eq"
                                    : row.kind === "chg"
                                      ? "diff-chg"
                                      : row.kind === "add"
                                        ? "diff-add"
                                        : "diff-del"
                                }`}
                              >
                                {row.right || "-"}
                              </div>
                            </Fragment>
                          ))}
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                    {translation.error ?? "Translation failed."}
                  </p>
                )}
              </section>
            ) : null}

            {translation?.success ? (
              <section className="card p-5 md:p-6">
                <StepHeader
                  step="Step 4"
                  title="Export Clean DOCX"
                  subtitle="Uses your optional template if provided, otherwise fixed structured output."
                />
                <button className="btn-primary min-w-[220px]" type="button" onClick={onGenerateDoc} disabled={generating}>
                  {generating ? "Generating DOCX..." : "Download Translated COA (.docx)"}
                </button>
              </section>
            ) : null}

            {errorMessage ? (
              <section className="rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {errorMessage}
              </section>
            ) : null}
          </section>
        </div>
      </main>
    </>
  );
}
