
import sys
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_streamlit():
    # If your app entry is streamlit_frontend.py or app.py (update as needed)
    import subprocess
    script_candidates = ["src/streamlit_frontend.py", "src/app.py", "app.py"]
    for s in script_candidates:
        if Path(s).exists():
            target = s
            break
    else:
        logger.error("No Streamlit entry file found (src/streamlit_frontend.py or src/app.py).")
        return

    logger.info("Launching Streamlit: %s", target)
    subprocess.run(["streamlit", "run", target])

def run_cli_single_file(file_path: str):
    # Lightweight pipeline: parse -> validate -> analyze -> generate reports
    from src.document_parser import ADGMDocumentParser
    from src.knowledge_base import ADGMKnowledgeBase
    from src.compliance_validator import ADGMComplianceValidator
    from src.legal_analyzer import LegalAnalyzer
    from src.report_generator import ReportGenerator

    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    logger.info("Running single-file validation pipeline for: %s", file_path)
    parser = ADGMDocumentParser()
    kb = ADGMKnowledgeBase()
    # if KB empty, attempt to build from defaults (best-effort)
    try:
        if kb.get_status().get("status") == "empty":
            kb.build_knowledge_base()
    except Exception:
        pass

    validator = ADGMComplianceValidator(kb)
    analyzer = LegalAnalyzer(kb)
    generator = ReportGenerator(output_dir="reports")

    # parse
    doc = parser.parse_uploaded_file(open(file_path, "rb"))
    # validate (single doc returns ComplianceReport object)
    report_obj = validator.validate_document(doc)
    # make aggregate dict in same shape as validate_all_documents result
    agg = {
        "process": (doc.metadata.document_type or "Unknown").title(),
        "documents_uploaded": 1,
        "required_documents": 0,
        "missing_documents": report_obj.missing_clauses or [],
        "issues_found": [
            {
                "document": i.document_name,
                "section": i.section,
                "issue": i.description,
                "severity": i.severity.value,
                "suggestion": i.suggestion,
                "adgm_reference": i.adgm_reference,
                "confidence": i.confidence
            } for i in report_obj.issues
        ],
        "per_document_reports": [
            {
                "document_name": report_obj.document_name,
                "document_type": report_obj.document_type,
                "overall_score": report_obj.overall_score,
                "processing_date": report_obj.processing_date,
                "validation_summary": report_obj.validation_summary,
                "recommendations": report_obj.recommendations,
                "missing_clauses": report_obj.missing_clauses
            }
        ]
    }

    # legal analysis
    legal_report = analyzer.analyze_document(doc)
    # attach legal flags to aggregate (optional)
    for f in legal_report.flags:
        agg["issues_found"].append({
            "document": f.document_name,
            "section": f.section,
            "issue": f.description,
            "severity": f.severity,
            "suggestion": f.suggestion,
            "kb_reference": f.kb_reference.to_dict() if f.kb_reference else None,
            "confidence": f.confidence
        })

    outputs = generator.generate_reports(agg, prefix=file_path.stem)
    logger.info("Reports generated: %s", outputs)
    print("Reports:", outputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="streamlit", choices=["streamlit", "cli"])
    parser.add_argument("--file", "-f", help="Path to a single docx/pdf file for CLI mode")
    args = parser.parse_args()

    if args.mode == "streamlit":
        run_streamlit()
    else:
        if not args.file:
            print("Provide --file for CLI mode. Example: python -m src.main cli --file samples/example.docx")
            sys.exit(1)
        run_cli_single_file(args.file)

if __name__ == "__main__":
    main()
 
