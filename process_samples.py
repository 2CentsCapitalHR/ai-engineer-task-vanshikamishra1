

import logging
from pathlib import Path
from src.document_parser import ADGMDocumentParser
from src.knowledge_base import ADGMKnowledgeBase
from src.compliance_validator import ADGMComplianceValidator
from src.legal_analyzer import LegalAnalyzer
from src.document_editor import DocumentEditor
from src.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    samples_dir = Path("samples")
    if not samples_dir.exists():
        logger.error("samples/ directory not found. Run make_samples.py first.")
        return

    # Components
    kb = ADGMKnowledgeBase(db_dir="data/adgm_kb")
    # If KB empty, attempt to build by default (best-effort)
    try:
        if kb.get_status().get("status") != "complete":
            kb.build_knowledge_base()
    except Exception:
        logger.exception("KB build attempt failed; continuing with empty KB.")

    parser = ADGMDocumentParser()
    validator = ADGMComplianceValidator(kb)
    analyzer = LegalAnalyzer(kb)
    editor = DocumentEditor(output_dir="annotated_docs")
    report_gen = ReportGenerator(output_dir="reports")

    docs = []
    for f in samples_dir.glob("*.docx"):
        with open(f, "rb") as fh:
            parsed = parser.parse_uploaded_file(fh)
            docs.append(parsed)

    if not docs:
        logger.error("No sample docs parsed; exiting.")
        return

    # aggregate validation
    agg = validator.validate_all_documents(docs)
    # attach legal flags per doc
    legal_reports = [analyzer.analyze_document(d) for d in docs]

    # create annotated docs
    for idx, d in enumerate(docs):
        combined_issues = []
        # per-document compliance missing clauses
        per_report = agg["per_document_reports"][idx]
        for m in per_report.get("missing_clauses", []):
            combined_issues.append({
                "section": "Mandatory Clause",
                "issue": f"Missing clause: {m}",
                "suggestion": f"Add clause: {m}",
                "severity": "MEDIUM"
            })
        # legal flags
        for flag in legal_reports[idx].flags:
            combined_issues.append({
                "section": flag.section,
                "issue": f"{flag.type}: {flag.description}",
                "suggestion": flag.suggestion or "",
                "severity": flag.severity.upper()
            })
        # save uploaded file bytes to temp path (we have original sample path)
        original_path = Path("samples") / d.metadata.filename
        annotated_path = editor.annotate_document(str(original_path), combined_issues)
        logger.info("Annotated file saved: %s", annotated_path)

    # generate comprehensive reports
    out = report_gen.generate_reports(agg, prefix="samples_aggregate")
    logger.info("Reports created: %s", out)
    print("Done. Reports:", out)

if __name__ == "__main__":
    main()
