"""
Predefined COA (Certificate of Analysis) document structure.

Defines the fixed sections that every translated COA document must contain.
The translator maps extracted text into these fields via structured JSON output,
and the doc_generator populates them into the Word document in order.
"""

# Each entry: (field_key, russian_label, description_for_prompt, is_table)
# The order here determines the order in the output Word document.
COA_SECTIONS = [
    (
        "document_title",
        "Наименование документа",
        "The document title, e.g. 'Certificate of Analysis'",
        False,
    ),
    (
        "company_info",
        "Информация о компании",
        "Manufacturer/supplier company name, address, logo text, contact info",
        False,
    ),
    (
        "product_name",
        "Наименование продукта",
        "Product name, trade name, INN/generic name",
        False,
    ),
    (
        "product_details",
        "Сведения о продукте",
        "CAS number, molecular formula, molecular weight, structural description, "
        "grade, pharmacopoeia reference, dosage form",
        False,
    ),
    (
        "batch_info",
        "Информация о серии",
        "Batch/Lot number, manufacturing date, expiry/retest date, batch size, "
        "package configuration",
        False,
    ),
    (
        "storage_conditions",
        "Условия хранения",
        "Storage conditions, temperature requirements, special precautions "
        "(protect from light, moisture, etc.)",
        False,
    ),
    (
        "test_results",
        "Результаты испытаний",
        "The main analytical results table with columns: Test/Parameter, "
        "Method, Acceptance Criteria/Specification, Result. "
        "This is typically the largest section of a COA. "
        "Include ALL tests: appearance, identification, assay, purity, "
        "impurities, water content, residual solvents, heavy metals, "
        "dissolution, microbial limits, endotoxins, etc. "
        "Return this as a list of rows, each row being a list of cell values.",
        True,
    ),
    (
        "conclusion",
        "Заключение",
        "Overall conclusion/disposition statement, e.g. 'The product complies "
        "with the specification', release decision",
        False,
    ),
    (
        "signatures",
        "Подписи",
        "Authorized signatory names, titles, QC/QA approval, dates of "
        "approval/release",
        False,
    ),
    (
        "notes",
        "Примечания",
        "Any additional notes, footnotes, legends, abbreviation explanations, "
        "or supplementary information",
        False,
    ),
]

# Field keys in order
COA_FIELD_KEYS = [s[0] for s in COA_SECTIONS]

# Mapping: field_key -> russian label
COA_FIELD_LABELS = {s[0]: s[1] for s in COA_SECTIONS}

# Mapping: field_key -> is_table
COA_FIELD_IS_TABLE = {s[0]: s[3] for s in COA_SECTIONS}


def get_section_descriptions_for_prompt() -> str:
    """
    Build a description of each section for inclusion in the translation
    prompt, so the LLM knows exactly what to put in each field.
    """
    lines = []
    for key, label, desc, is_table in COA_SECTIONS:
        type_hint = "TABLE (list of rows)" if is_table else "TEXT (string)"
        lines.append(f'  "{key}" ({label}) [{type_hint}]: {desc}')
    return "\n".join(lines)
