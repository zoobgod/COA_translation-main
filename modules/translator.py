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

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GLOSSARY_RULES = """\
MANDATORY PHARMACEUTICAL GLOSSARY (English → Russian) — always prefer these \
over any generic translation:
{glossary}
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

STRUCTURED_SYSTEM_PROMPT = """\
You are a professional pharmaceutical translator specialising in Certificate \
of Analysis (COA) documents, English → Russian.

{common_rules}

{glossary_section}

OUTPUT FORMAT — you MUST return **valid JSON only** (no markdown fences, no \
commentary) with the following keys. Every key must be present; use an empty \
string "" or empty list [] if the source document does not contain information \
for that section.

Section definitions:
{section_descriptions}

JSON schema:
{{
{json_keys}
}}

For "test_results" (the table), return a JSON array of arrays. The first \
inner array is the header row. Example:
  "test_results": [
    ["Испытание", "Метод", "Критерии приемлемости", "Результат"],
    ["Внешний вид", "Визуальный", "Белый порошок", "Соответствует"],
    ...
  ]

For all other keys return a plain Russian-language string.
Do not omit sections or shorten long tables; preserve full COA content.
"""

PLAIN_SYSTEM_PROMPT = """\
You are a professional pharmaceutical translator specialising in translating \
Certificate of Analysis (COA) documents from English to Russian.

{common_rules}

{glossary_section}

Output ONLY the translated text — no JSON, no markdown fences, no commentary.
Preserve the original document layout as closely as possible.
Preserve any table structure using | as the column delimiter.
"""


def _build_system_prompt(structured: bool) -> str:
    glossary_text = get_glossary_prompt_section()
    glossary_section = _GLOSSARY_RULES.format(glossary=glossary_text)
    common_rules = _COMMON_RULES

    if structured:
        section_descriptions = get_section_descriptions_for_prompt()
        json_keys = ",\n".join(f'  "{k}": "..."' for k in COA_FIELD_KEYS)
        return STRUCTURED_SYSTEM_PROMPT.format(
            common_rules=common_rules,
            glossary_section=glossary_section,
            section_descriptions=section_descriptions,
            json_keys=json_keys,
        )
    else:
        return PLAIN_SYSTEM_PROMPT.format(
            common_rules=common_rules,
            glossary_section=glossary_section,
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
) -> dict:
    """
    Translate pharmaceutical COA text from English to Russian using OpenAI.

    Returns a **plain** translation (single text string) suitable for preview
    and for the legacy document-generation path.
    """
    return _translate_plain(text, api_key, model, progress_callback)


def translate_text_structured(
    text: str,
    api_key: str,
    model: str = "gpt-4o",
    progress_callback: Optional[callable] = None,
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
    return _translate_structured(text, api_key, model, progress_callback)


# ---------------------------------------------------------------------------
# Internal — plain mode
# ---------------------------------------------------------------------------

def _translate_plain(
    text: str,
    api_key: str,
    model: str,
    progress_callback: Optional[callable],
) -> dict:
    if not text.strip():
        return _error_result("No text provided for translation", model)

    try:
        client = OpenAI(api_key=api_key)
        system_prompt = _build_system_prompt(structured=False)
        chunks = _chunk_text(text)
        translated_parts: list[str] = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            user_message = (
                "Translate the following pharmaceutical COA text from English "
                "to Russian. Output ONLY the translation, nothing else.\n\n"
                + chunk
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
            )

            translated = response.choices[0].message.content
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
) -> dict:
    if not text.strip():
        return _error_result("No text provided for translation", model)

    try:
        client = OpenAI(api_key=api_key)
        system_prompt = _build_system_prompt(structured=True)

        if progress_callback:
            progress_callback(1, 2)

        user_message = (
            "Below is the full extracted text of a pharmaceutical Certificate "
            "of Analysis (COA). Translate it to Russian and map the content "
            "into the predefined JSON structure described in your instructions. "
            "Return ONLY valid JSON.\n\n"
            + text
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            # Encourage the model to produce enough output for the full COA
            max_tokens=STRUCTURED_MAX_TOKENS,
        )

        raw = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        if progress_callback:
            progress_callback(2, 2)

        # Parse JSON — strip markdown code fences if the model adds them
        json_str = raw.strip()
        if json_str.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        sections = _normalise_sections(json.loads(json_str))

        # If structured output is likely truncated/incomplete, backfill full
        # plain translation into notes so no source content is lost.
        if _needs_plain_backfill(text, sections, finish_reason):
            logger.warning(
                "Structured output may be incomplete (finish_reason=%s); "
                "running plain translation backfill",
                finish_reason,
            )
            plain_result = _translate_plain(text, api_key, model, None)
            if plain_result["success"] and plain_result["translated_text"].strip():
                merged_notes = _merge_notes(
                    sections.get("notes", ""),
                    plain_result["translated_text"],
                )
                sections["notes"] = merged_notes
            else:
                logger.warning("Plain-translation backfill failed")

        return {
            "sections": sections,
            "translated_text": _build_preview_from_sections(sections),
            "success": True,
            "error": None,
            "model_used": model,
            "chunks_translated": 1,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse structured translation JSON: {e}")
        # Fall back to plain translation
        logger.info("Falling back to plain translation mode")
        plain_result = _translate_plain(text, api_key, model, progress_callback)
        if plain_result["success"]:
            # Build a best-effort sections dict with all content in notes
            sections = {k: "" for k in COA_FIELD_KEYS}
            sections["notes"] = plain_result["translated_text"]
            plain_result["sections"] = sections
        return plain_result

    except Exception as e:
        logger.error(f"Structured translation failed: {e}")
        return _error_result(str(e), model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(error: str, model: str) -> dict:
    return {
        "translated_text": "",
        "sections": {},
        "success": False,
        "error": error,
        "model_used": model,
        "chunks_translated": 0,
    }


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
