"""
OpenAI-based translation module for pharmaceutical COA documents.

Translates English text to Russian with pharmaceutical-specific context
and glossary enforcement.

Supports two output modes:
    - **structured**: Returns a dict mapping predefined COA section keys
      to translated Russian content (used for fixed-structure Word output).
    - **plain**: Returns a single translated text string (legacy / preview).
"""

import json
import logging
from typing import Optional

from openai import OpenAI

from modules.glossary import get_glossary_prompt_section
from modules.coa_structure import (
    COA_FIELD_KEYS,
    COA_FIELD_LABELS,
    get_section_descriptions_for_prompt,
)

logger = logging.getLogger(__name__)

# Maximum characters per translation chunk (to stay within token limits)
MAX_CHUNK_SIZE = 6000
STRUCTURED_MAX_TOKENS = 12000
STRUCTURED_MIN_ALNUM_RATIO = 0.45
CUSTOM_GLOSSARY_MAX_CHARS = 24000

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GLOSSARY_RULES = """\
MANDATORY PHARMACEUTICAL GLOSSARY (English → Russian) — always prefer these \
over any generic translation:
{glossary}
"""

_USER_GLOSSARY_RULES = """\
USER-PROVIDED GLOSSARY (highest priority; override defaults when conflicts exist):
{user_glossary}
"""

_COMMON_RULES = """\
Translation rules (apply to ALL output):
1. Translate ALL English text to Russian.
2. Keep numerical values, chemical formulas, CAS numbers, and catalog numbers UNCHANGED.
3. Keep Latin scientific names in their original Latin form.
4. Use the pharmaceutical glossary below for standard terminology — these \
translations are mandatory.
5. Maintain standard Russian pharmaceutical terminology consistent with the \
Russian Pharmacopoeia (Государственная Фармакопея).
6. Keep internationally-recognised abbreviations (pH, HPLC, GC, etc.) but \
provide the Russian equivalent from the glossary in parentheses where it \
first appears.
7. Do NOT add explanations, comments, or notes of your own — translate only.
8. NEVER summarize or compress content; preserve all meaningful lines, \
table rows, and metadata from the source.
"""

PLAIN_SYSTEM_PROMPT = """\
You are a professional pharmaceutical translator specialising in translating \
Certificate of Analysis (COA) documents from English to Russian.

{common_rules}

{glossary_section}

Output ONLY the translated text — no JSON, no markdown fences, no commentary.
Preserve the original document layout as closely as possible.
Preserve any table structure using | as the column delimiter.
Do not omit any lines, rows, limits, footnotes, or acceptance criteria.
"""

STRUCTURE_MAPPING_SYSTEM_PROMPT = """\
You are a pharmaceutical document structuring expert.

You will receive:
1) A FULL Russian translation of a COA.
2) Optional supplemental extracted test tables.

Your task is to map content into the predefined JSON structure.
Do not summarize. Preserve complete test data in "test_results".

OUTPUT FORMAT — return valid JSON only with these keys:
{{
{json_keys},
  "template_fields": {{}},
  "template_heading_map": {{}}
}}

Section definitions:
{section_descriptions}

Rules:
- Keep all meaningful content.
- For "test_results" return JSON array of arrays.
- For text fields return plain strings.
- If template hints are provided, populate "template_fields" and
  "template_heading_map" accordingly.
"""


def _build_system_prompt(structured: bool, custom_glossary: str = "") -> str:
    glossary_text = _build_combined_glossary(custom_glossary)
    glossary_section = _GLOSSARY_RULES.format(glossary=glossary_text)
    common_rules = _COMMON_RULES

    if structured:
        return _build_structuring_prompt()
    return PLAIN_SYSTEM_PROMPT.format(
        common_rules=common_rules,
        glossary_section=glossary_section,
    )


def _build_structuring_prompt() -> str:
    section_descriptions = get_section_descriptions_for_prompt()
    json_keys = ",\n".join(f'  "{k}": "..."' for k in COA_FIELD_KEYS)
    return STRUCTURE_MAPPING_SYSTEM_PROMPT.format(
        section_descriptions=section_descriptions,
        json_keys=json_keys,
    )


# ---------------------------------------------------------------------------
# Chunking (for plain mode on very large documents)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split text into chunks respecting paragraph boundaries."""
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            if len(para) > max_size:
                lines = para.split("\n")
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = line + "\n"
                    else:
                        current_chunk += line + "\n"
            else:
                current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_text(
    text: str,
    api_key: str,
    model: str = "gpt-4o",
    progress_callback: Optional[callable] = None,
    custom_glossary: str = "",
) -> dict:
    """
    Translate pharmaceutical COA text from English to Russian using OpenAI.

    Returns a **plain** translation (single text string) suitable for preview
    and for the legacy document-generation path.
    """
    return _translate_plain(
        text,
        api_key,
        model,
        progress_callback,
        custom_glossary=custom_glossary,
    )


def translate_text_structured(
    text: str,
    api_key: str,
    model: str = "gpt-4o",
    progress_callback: Optional[callable] = None,
    template_hints: Optional[dict] = None,
    table_supplement: str = "",
    custom_glossary: str = "",
) -> dict:
    """
    Translate pharmaceutical COA text and return **structured** output — a
    dict keyed by the predefined COA section keys with Russian values.

    Returns:
        dict with keys:
            - 'sections': dict mapping COA field keys to translated values
            - 'translated_text': flattened plain-text version for preview
            - 'success' / 'error' / 'model_used' / 'chunks_translated'
    """
    return _translate_structured(
        text,
        api_key,
        model,
        progress_callback,
        template_hints=template_hints,
        table_supplement=table_supplement,
        custom_glossary=custom_glossary,
    )


# ---------------------------------------------------------------------------
# Internal — plain mode
# ---------------------------------------------------------------------------

def _translate_plain(
    text: str,
    api_key: str,
    model: str,
    progress_callback: Optional[callable],
    custom_glossary: str = "",
) -> dict:
    if not text.strip():
        return _error_result("No text provided for translation", model)

    try:
        client = OpenAI(api_key=api_key)
        system_prompt = _build_system_prompt(
            structured=False,
            custom_glossary=custom_glossary,
        )
        chunks = _chunk_text(text)
        translated_parts: list[str] = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            user_message = (
                "Translate the following pharmaceutical COA text from English "
                "to Russian. Output ONLY the translation, nothing else. "
                "Do not omit any lines or table rows.\n\n"
                + chunk
            )

            response = _create_chat_completion(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
            )

            translated = response["content"]
            if translated:
                translated_parts.append(translated.strip())

        full_translation = "\n\n".join(translated_parts)
        return {
            "translated_text": full_translation,
            "success": True,
            "error": None,
            "model_used": model,
            "chunks_translated": len(chunks),
        }

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return _error_result(str(e), model)


# ---------------------------------------------------------------------------
# Internal — structured mode
# ---------------------------------------------------------------------------

def _translate_structured(
    text: str,
    api_key: str,
    model: str,
    progress_callback: Optional[callable],
    template_hints: Optional[dict] = None,
    table_supplement: str = "",
    custom_glossary: str = "",
) -> dict:
    if not text.strip():
        return _error_result("No text provided for translation", model)

    try:
        if progress_callback:
            progress_callback(1, 3)

        # Pass 1: high-fidelity full translation (quality-first).
        plain_result = _translate_plain(
            text,
            api_key,
            model,
            None,
            custom_glossary=custom_glossary,
        )
        if not plain_result["success"]:
            return plain_result

        full_translation = plain_result["translated_text"].strip()
        if not full_translation:
            return _error_result("Empty translation result", model)

        if progress_callback:
            progress_callback(2, 3)

        # Pass 2: map translated content to structured sections.
        sections, template_fields, template_heading_map, finish_reason = (
            _structure_translated_content(
                translated_text=full_translation,
                table_supplement=table_supplement,
                api_key=api_key,
                model=model,
                template_hints=template_hints,
            )
        )

        if _needs_plain_backfill(text, sections, finish_reason):
            sections["notes"] = _merge_notes(sections.get("notes", ""), full_translation)

        if progress_callback:
            progress_callback(3, 3)

        return {
            "sections": sections,
            "template_fields": template_fields,
            "template_heading_map": template_heading_map,
            # Show full translation in UI preview (not only section summary)
            "translated_text": full_translation,
            "success": True,
            "error": None,
            "model_used": model,
            "chunks_translated": plain_result.get("chunks_translated", 1),
        }

    except Exception as e:
        logger.error(f"Structured translation failed: {e}")

        # Final fallback: still provide clean full translation for end users.
        plain_result = _translate_plain(
            text,
            api_key,
            model,
            None,
            custom_glossary=custom_glossary,
        )
        if plain_result["success"]:
            sections = {k: "" for k in COA_FIELD_KEYS}
            sections["notes"] = plain_result["translated_text"]
            plain_result["sections"] = sections
            plain_result["template_fields"] = _normalise_template_fields(
                None,
                template_hints,
                sections,
            )
            plain_result["template_heading_map"] = _normalise_template_heading_map(
                None,
                template_hints,
            )
            return plain_result

        return _error_result(str(e), model)


def _build_combined_glossary(custom_glossary: str = "") -> str:
    """Merge built-in glossary with optional user-provided glossary."""
    base = get_glossary_prompt_section().strip()
    user = (custom_glossary or "").strip()
    if not user:
        return base

    if len(user) > CUSTOM_GLOSSARY_MAX_CHARS:
        logger.warning(
            "Custom glossary too large (%s chars), truncating to %s chars",
            len(user),
            CUSTOM_GLOSSARY_MAX_CHARS,
        )
        user = user[:CUSTOM_GLOSSARY_MAX_CHARS]

    user_section = _USER_GLOSSARY_RULES.format(user_glossary=user)
    return f"{base}\n\n{user_section}"


def _structure_translated_content(
    translated_text: str,
    table_supplement: str,
    api_key: str,
    model: str,
    template_hints: Optional[dict],
) -> tuple[dict, dict, dict, str]:
    """
    Map already-translated Russian COA text into structured sections.
    Returns (sections, template_fields, template_heading_map, finish_reason).
    """
    client = OpenAI(api_key=api_key)
    system_prompt = _build_structuring_prompt()

    user_message = (
        "Below is the FULL Russian translation of a COA. "
        "Map it to the requested JSON structure.\n\n"
        "=== FULL RUSSIAN TRANSLATION ===\n"
        f"{translated_text}"
    )

    if table_supplement.strip():
        user_message += (
            "\n\n=== SUPPLEMENTAL EXTRACTED TABLES (source-side recovery) ===\n"
            f"{table_supplement}"
            "\nUse this supplement primarily to complete test_results rows."
        )

    template_instruction = _build_template_instruction(template_hints)
    if template_instruction:
        user_message += "\n\n" + template_instruction

    response = _create_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=STRUCTURED_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    finish_reason = response.get("finish_reason") or "stop"
    parsed = json.loads(_strip_json_fences(response.get("content", "")))
    parsed_dict = parsed if isinstance(parsed, dict) else {}

    sections = _normalise_sections(parsed_dict)
    template_fields = _normalise_template_fields(
        parsed_dict.get("template_fields"),
        template_hints,
        sections,
    )
    template_heading_map = _normalise_template_heading_map(
        parsed_dict.get("template_heading_map"),
        template_hints,
    )
    return sections, template_fields, template_heading_map, finish_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(error: str, model: str) -> dict:
    return {
        "translated_text": "",
        "sections": {},
        "template_fields": {},
        "template_heading_map": {},
        "success": False,
        "error": error,
        "model_used": model,
        "chunks_translated": 0,
    }


def _strip_json_fences(raw: str) -> str:
    """Remove markdown code fences if present."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


def _create_chat_completion(client: OpenAI, model: str, **kwargs) -> dict:
    """
    Wrapper around chat.completions that:
    - adapts token-parameter naming by model family
    - retries without response_format when unsupported
    - falls back to Responses API for models not accepted by Chat Completions
    """
    payload = dict(kwargs)
    max_tokens = payload.pop("max_tokens", None)
    if max_tokens is not None:
        token_param = (
            "max_completion_tokens"
            if _uses_completion_token_param(model)
            else "max_tokens"
        )
        payload[token_param] = max_tokens

    def _call_chat(chat_payload: dict):
        response = client.chat.completions.create(model=model, **chat_payload)
        return {
            "content": response.choices[0].message.content or "",
            "finish_reason": response.choices[0].finish_reason or "stop",
        }

    try:
        return _call_chat(payload)
    except Exception as e:
        message = str(e).lower()
        if "temperature" in payload and _is_temperature_error(message):
            logger.warning("temperature unsupported, retrying without it: %s", e)
            reduced = dict(payload)
            reduced.pop("temperature", None)
            return _call_chat(reduced)
        if "response_format" in payload and "response_format" in message:
            logger.warning("response_format unsupported, retrying without it: %s", e)
            reduced = dict(payload)
            reduced.pop("response_format", None)
            try:
                return _call_chat(reduced)
            except Exception as inner:
                inner_message = str(inner).lower()
                if "temperature" in reduced and _is_temperature_error(inner_message):
                    reduced2 = dict(reduced)
                    reduced2.pop("temperature", None)
                    try:
                        return _call_chat(reduced2)
                    except Exception as inner2:
                        inner_message = str(inner2).lower()
                if _should_try_responses_fallback(model, inner_message):
                    return _create_with_responses_api(client, model, reduced)
                raise

        if _should_try_responses_fallback(model, message):
            logger.warning(
                "Chat completion failed for model '%s'; trying Responses API fallback",
                model,
            )
            return _create_with_responses_api(client, model, payload)
        raise


def _should_try_responses_fallback(model: str, error_message: str) -> bool:
    m = (model or "").lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o")
        or "responses api" in error_message
        or "not supported in the v1/chat/completions endpoint" in error_message
        or "unsupported model" in error_message
    )


def _create_with_responses_api(client: OpenAI, model: str, payload: dict) -> dict:
    """
    Best-effort fallback using Responses API for newer models.
    """
    if not hasattr(client, "responses"):
        raise RuntimeError("Installed OpenAI SDK does not support Responses API")

    req: dict = {
        "model": model,
        "input": payload.get("messages", []),
    }

    if "temperature" in payload:
        req["temperature"] = payload["temperature"]

    if "max_completion_tokens" in payload:
        req["max_output_tokens"] = payload["max_completion_tokens"]
    elif "max_tokens" in payload:
        req["max_output_tokens"] = payload["max_tokens"]

    # Keep prompt-level JSON constraints; omit chat-only response_format here.
    try:
        response = client.responses.create(**req)
    except Exception as e:
        if "temperature" in req and _is_temperature_error(str(e).lower()):
            req2 = dict(req)
            req2.pop("temperature", None)
            response = client.responses.create(**req2)
        else:
            raise
    content = getattr(response, "output_text", "") or ""
    finish = getattr(response, "status", None) or "stop"
    return {"content": content, "finish_reason": finish}


def _uses_completion_token_param(model: str) -> bool:
    """Models that typically expect max_completion_tokens."""
    m = (model or "").lower()
    return m.startswith("o") or m.startswith("gpt-5")


def _is_temperature_error(message: str) -> bool:
    return (
        "temperature" in message
        and ("not supported" in message or "unsupported" in message)
    )


def _build_template_instruction(template_hints: Optional[dict]) -> str:
    """
    Build prompt instructions that align translation output with a user
    template in the same API call.
    """
    if not isinstance(template_hints, dict):
        return ""

    placeholders = template_hints.get("placeholders") or []
    headings = template_hints.get("headings") or []

    lines: list[str] = []
    if placeholders:
        lines.append(
            "Template placeholders were supplied by the user. Include an "
            "additional JSON object key named \"template_fields\" where each "
            "placeholder below is present as a key with translated content "
            "(or empty string if unavailable):"
        )
        lines.extend(f'- "{p}"' for p in placeholders[:80])
        lines.append(
            'Example: "template_fields": {"product": "...", "batch_no": "..."}'
        )

    if headings:
        lines.append(
            "Template heading hints. Also include JSON key "
            "\"template_heading_map\" mapping each heading to one of these "
            f"section keys: {', '.join(COA_FIELD_KEYS)}."
        )
        lines.append(
            "Example: "
            '"template_heading_map": {"Product": "product_name", '
            '"Results": "test_results"}'
        )
        lines.extend(f"- {h}" for h in headings[:30])

    return "\n".join(lines)


def _normalise_template_fields(
    template_fields_raw,
    template_hints: Optional[dict],
    sections: dict,
) -> dict:
    """
    Ensure template_fields is always a dict and contains all provided
    placeholders (if any), with heuristic fallback from section values.
    """
    result: dict = {}
    if isinstance(template_fields_raw, dict):
        for key, value in template_fields_raw.items():
            if key:
                result[str(key)] = "" if value is None else str(value)

    placeholders = []
    if isinstance(template_hints, dict):
        placeholders = template_hints.get("placeholders") or []

    for placeholder in placeholders:
        if placeholder in result and result[placeholder].strip():
            continue
        mapped_key = _map_placeholder_to_section(placeholder)
        if not mapped_key:
            result.setdefault(placeholder, "")
            continue

        value = sections.get(mapped_key, "")
        if isinstance(value, list):
            lines = [" | ".join(str(cell) for cell in row) for row in value]
            result[placeholder] = "\n".join(lines)
        else:
            result[placeholder] = "" if value is None else str(value)

    return result


def _map_placeholder_to_section(placeholder: str) -> Optional[str]:
    """Heuristic mapping from arbitrary placeholder names to COA keys."""
    raw = (placeholder or "").strip().lower().replace("_", " ")
    cleaned = " ".join(raw.split())
    if not cleaned:
        return None

    if placeholder in COA_FIELD_KEYS:
        return placeholder

    candidate_space = []
    for key in COA_FIELD_KEYS:
        candidate_space.append((key, key.replace("_", " ")))
        candidate_space.append((key, COA_FIELD_LABELS[key].lower()))

    best_key = None
    best_score = 0.0
    from difflib import SequenceMatcher

    for key, label in candidate_space:
        score = SequenceMatcher(a=cleaned, b=label).ratio()
        if label in cleaned or cleaned in label:
            score = max(score, 0.9)
        if score > best_score:
            best_key = key
            best_score = score

    return best_key if best_score >= 0.62 else None


def _normalise_template_heading_map(
    heading_map_raw,
    template_hints: Optional[dict],
) -> dict:
    """Validate/normalise AI heading map for template insertion."""
    hints = template_hints if isinstance(template_hints, dict) else {}
    headings = hints.get("headings") or []
    normalised: dict = {}

    if isinstance(heading_map_raw, dict):
        for heading, key in heading_map_raw.items():
            if not heading or not key:
                continue
            mapped = str(key).strip()
            if mapped in COA_FIELD_KEYS:
                normalised[str(heading)] = mapped

    # Fallback heuristic mapping when model omitted heading map.
    for heading in headings:
        if heading in normalised:
            continue
        mapped = _map_placeholder_to_section(heading)
        if mapped:
            normalised[heading] = mapped

    return normalised


def _normalise_sections(sections: dict) -> dict:
    """Ensure consistent section types and presence of all expected keys."""
    if not isinstance(sections, dict):
        sections = {}

    normalised: dict = {}

    for key in COA_FIELD_KEYS:
        value = sections.get(key, "" if key != "test_results" else [])

        if key == "test_results":
            if isinstance(value, list):
                table_rows: list[list[str]] = []
                for row in value:
                    if isinstance(row, list):
                        table_rows.append([str(cell) for cell in row])
                normalised[key] = table_rows
            elif isinstance(value, str) and value.strip():
                normalised[key] = [
                    [cell.strip() for cell in line.split("|")]
                    for line in value.splitlines()
                    if line.strip()
                ]
            else:
                normalised[key] = []
        else:
            if value is None:
                normalised[key] = ""
            elif isinstance(value, list):
                normalised[key] = "\n".join(str(item) for item in value if item)
            else:
                normalised[key] = str(value)

    return normalised


def _build_preview_from_sections(sections: dict) -> str:
    """Build a plain-text preview from structured section data."""
    preview_parts: list[str] = []
    for key in COA_FIELD_KEYS:
        label = COA_FIELD_LABELS[key]
        value = sections.get(key, "")

        if isinstance(value, list):
            if not value:
                continue
            rows = [" | ".join(str(c) for c in row) for row in value]
            preview_parts.append(f"[{label}]\n" + "\n".join(rows))
        elif isinstance(value, str) and value.strip():
            preview_parts.append(f"[{label}]\n{value}")

    return "\n\n".join(preview_parts)


def _needs_plain_backfill(source_text: str, sections: dict, finish_reason: str) -> bool:
    """
    Decide whether structured translation likely omitted content and needs
    plain-translation fallback appended to notes.
    """
    if finish_reason == "length":
        return True

    source_alnum = sum(1 for ch in source_text if ch.isalnum())
    if source_alnum < 300:
        return False

    structured_text = _build_preview_from_sections(sections)
    structured_alnum = sum(1 for ch in structured_text if ch.isalnum())
    ratio = structured_alnum / source_alnum if source_alnum else 1.0
    return ratio < STRUCTURED_MIN_ALNUM_RATIO


def _merge_notes(existing_notes: str, plain_translation: str) -> str:
    """Append full plain translation to notes while avoiding duplicates."""
    plain = plain_translation.strip()
    if not plain:
        return existing_notes

    note_header = "Полный перевод (резервный слой для проверки полноты):"
    if plain in existing_notes:
        return existing_notes

    existing = existing_notes.strip()
    if existing:
        return f"{existing}\n\n{note_header}\n{plain}"
    return f"{note_header}\n{plain}"
