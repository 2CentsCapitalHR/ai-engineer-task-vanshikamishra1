import io
import json
import logging
from pathlib import Path
import streamlit as st
from src.knowledge_base import ADGMKnowledgeBase
from src.compliance_validator import ADGMComplianceValidator
from src.legal_analyzer import LegalAnalyzer
from src.document_editor import DocumentEditor
from src.document_parser import ADGMDocumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_state():
    defaults = {
        "kb_built": False,
        "uploaded_files": [],
        "parsed_docs": [],
        "aggregate_report": None,
        "per_doc_reports": [],
        "legal_reports": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

@st.cache_resource
def get_components(kb_path="adgm_knowledge.index"):
    kb = ADGMKnowledgeBase(kb_path)
    validator = ADGMComplianceValidator(kb)
    analyzer = LegalAnalyzer(kb)
    editor = DocumentEditor(output_dir="annotated_docs")
    parser = ADGMDocumentParser()
    return kb, validator, analyzer, editor, parser

def header():
    st.set_page_config(page_title="ADGM Corporate Agent", layout="wide")
    st.markdown(
        """
        <div style="padding:12px;border-radius:8px;background:#f7f9fc">
            <h2 style="margin:0">ADGM Corporate Agent</h2>
            <div style="color:#6b7280">
                Document intelligence for ADGM incorporation & compliance — upload .docx, run validation, download annotated files and JSON reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sidebar(kb: ADGMKnowledgeBase):
    st.sidebar.header("Controls")
    with st.sidebar.expander("Knowledge Base"):
        if st.button("Build Knowledge Base"):
            build_kb(kb)
        if st.button("Clear Knowledge Base Cache"):
            try:
                Path(f"{kb.db_path}.faiss").unlink(missing_ok=True)
                Path(f"{kb.db_path}.json").unlink(missing_ok=True)
                st.success("Knowledge base cache cleared.")
            except Exception as e:
                st.error(f"Failed to clear KB: {e}")
    if st.sidebar.button("Run Aggregate Validation"):
        st.session_state.run_aggregate = True

def build_kb(kb: ADGMKnowledgeBase):
    with st.spinner("Building knowledge base..."):
        sources = [
            "Data Sources.pdf",
            "data/Data Sources.pdf",
            "data_sources/Data Sources.pdf",
            "resources/Data Sources.pdf",
        ]
        added = 0
        for p in sources:
            if Path(p).exists():
                kb.add_document(p)
                added += 1
        st.session_state.kb_built = added > 0
        if added:
            st.success(f"Knowledge base built: {added} documents added.")
        else:
            st.info("No data source found locally.")

def upload_section(parser: ADGMDocumentParser):
    st.header("Document Upload & Processing")
    uploaded = st.file_uploader(
        "Upload ADGM legal documents (.docx)",
        type=["docx"],
        accept_multiple_files=True
    )
    if uploaded:
        tmp_path = Path("tmp_uploads")
        tmp_path.mkdir(exist_ok=True)
        parsed_docs = []
        for file in uploaded:
            file_path = tmp_path / file.name
            with open(file_path, "wb") as out_file:
                out_file.write(file.getbuffer())
            doc_content = parser.parse_uploaded_file(file)
            parsed_docs.append(doc_content)
        st.session_state.uploaded_files = uploaded
        st.session_state.parsed_docs = parsed_docs
        st.success(f"Processed {len(uploaded)} file(s).")

def results_section(validator: ADGMComplianceValidator, analyzer: LegalAnalyzer, editor: DocumentEditor):
    st.header("Validation & Analysis")
    if not st.session_state.parsed_docs:
        st.info("No parsed documents. Upload .docx files first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        rows = [{
            "Filename": d.metadata.filename,
            "Type": d.metadata.document_type,
            "Words": d.metadata.word_count,
            "Confidence": f"{d.metadata.confidence_score:.0%}"
        } for d in st.session_state.parsed_docs]
        st.table(rows)

    with col2:
        if st.button("Run Compliance & Legal Analysis"):
            with st.spinner("Running analysis..."):
                aggregate = validator.validate_all_documents(st.session_state.parsed_docs)
                st.session_state.aggregate_report = aggregate
                legal_reports = [analyzer.analyze_document(doc) for doc in st.session_state.parsed_docs]
                st.session_state.legal_reports = legal_reports
                st.success("Analysis complete.")

    if st.session_state.aggregate_report:
        agg = st.session_state.aggregate_report
        st.subheader("Aggregate Report")
        st.write(f"Process: {agg.get('process')}")
        st.write(f"Documents uploaded: {agg.get('documents_uploaded')} / {agg.get('required_documents')}")
        if agg.get("missing_documents"):
            st.error(f"Missing: {', '.join(agg['missing_documents'])}")
        st.dataframe(agg.get("issues_found", [])[:20])
        st.download_button(
            label="Download Aggregate JSON Report",
            data=json.dumps(agg, indent=2, ensure_ascii=False),
            file_name="aggregate_adgm_report.json",
            mime="application/json"
        )

    if st.session_state.aggregate_report and st.session_state.legal_reports:
        st.subheader("Per-document Analysis")
        for idx, doc in enumerate(st.session_state.parsed_docs):
            with st.expander(f"{doc.metadata.filename} — {doc.metadata.document_type}"):
                comp_reports = st.session_state.aggregate_report.get("per_document_reports", [])
                if idx < len(comp_reports):
                    pr = comp_reports[idx]
                    st.write("Compliance score:", pr.get("overall_score"))
                    st.write("Recommendations:")
                    for r in pr.get("recommendations", []):
                        st.write(f"- {r}")
                lr = st.session_state.legal_reports[idx]
                if lr.flags:
                    for flag in lr.flags:
                        st.markdown(f"- [{flag.severity.upper()}] {flag.type}: {flag.description}")
                        if flag.suggestion:
                            st.markdown(f"  Suggestion: {flag.suggestion}")
                else:
                    st.success("No legal flags detected.")
                if st.button(f"Generate Annotated File: {doc.metadata.filename}", key=f"annotate_{idx}"):
                    combined_issues = []
                    if idx < len(comp_reports):
                        for m in comp_reports[idx].get("missing_clauses", []):
                            combined_issues.append({
                                "section": "Mandatory Clause",
                                "issue": f"Missing clause: {m}",
                                "suggestion": f"Add clause: {m}",
                                "severity": "MEDIUM"
                            })
                    for f in lr.flags:
                        combined_issues.append({
                            "section": f.section,
                            "issue": f"{f.type}: {f.description}",
                            "suggestion": f.suggestion or "",
                            "severity": f.severity.upper()
                        })
                    uploaded_file = st.session_state.uploaded_files[idx]
                    fp = Path("tmp_uploads")
                    out_path = fp / uploaded_file.name
                    with open(out_path, "wb") as wf:
                        wf.write(uploaded_file.getbuffer())
                    annotated_path = editor.annotate_document(str(out_path), combined_issues)
                    with open(annotated_path, "rb") as f:
                        st.download_button(
                            label=f"Download Annotated: {Path(annotated_path).name}",
                            data=f.read(),
                            file_name=Path(annotated_path).name,
                            mime="application/octet-stream"
                        )

def main():
    init_state()
    header()
    kb, validator, analyzer, editor, parser = get_components()
    sidebar(kb)
    upload_section(parser)
    results_section(validator, analyzer, editor)

if __name__ == "__main__":
    main()
