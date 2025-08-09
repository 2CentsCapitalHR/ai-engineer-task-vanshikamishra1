# src/compliance_validator.py

import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .document_parser import DocumentContent
from .knowledge_base import ADGMKnowledgeBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CITATION_DATABASE = {
    "missing_jurisdiction": "Per ADGM Companies Regulations 2020, Art. 6(1)(b), the Articles must specify that the company is subject to the jurisdiction of the ADGM Courts.",
    "incorrect_jurisdiction": "ADGM guidance mandates exclusive jurisdiction of ADGM Courts. References to other courts like UAE Federal Courts are non-compliant.",
    "missing_clause_liability of members": "The ADGM checklist for Private Companies requires a 'Liability of Members' clause to be explicitly defined in the Articles of Association.",
    "missing_clause_general meetings": "ADGM Companies Regulations outline the requirements for conducting general meetings, which must be included in the company's Articles.",
    "missing_clause_ADGM": "The document must contain explicit references to 'Abu Dhabi Global Market' or 'ADGM' to establish its legal framework.",
    "insufficient_adgm_terms": "To ensure clarity and compliance, documents should incorporate standard ADGM terminology, such as 'ADGM Registrar' or 'FSRA'.",
    "missing_signatures": "ADGM's document execution policy requires clear signatory blocks for all relevant parties, including names, titles, and dates.",
    "default": "Refer to the official ADGM Companies Regulations and incorporation checklists for detailed guidance."
}

# --- Data models / enums ---
class IssueSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceIssue:
    issue_id: str
    document_name: str
    section: str
    issue_type: str
    description: str
    severity: IssueSeverity
    adgm_reference: Optional[str]
    suggestion: Optional[str]
    confidence: float = 0.75
    location: Optional[str] = None

@dataclass
class ComplianceReport:
    document_name: str
    document_type: str
    overall_score: float
    issues: List[ComplianceIssue] = field(default_factory=list)
    missing_clauses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    validation_summary: Dict[str, Any] = field(default_factory=dict)

# --- Static ADGM rules (baseline) ---
class ADGMComplianceRules:
    JURISDICTION_REQUIRED = [r"\bADGM\b", r"\bADGM Courts\b", r"Abu\s+Dhabi\s+Global\s+Market"]
    JURISDICTION_FORBIDDEN = [r"\bUAE Federal Court(s)?\b", r"\bDubai Courts?\b", r"\bAbu Dhabi Courts\b(?!.*Global)"]
    MANDATORY_CLAUSES = {
        "articles_of_association": ["company name", "registered office", "objects of the company", "share capital", "liability of members", "directors powers", "general meetings", "ADGM"],
        "memorandum_of_association": ["company name", "registered office", "objects clause", "authorized share capital", "subscriber", "liability limitation"],
    }
    ADGM_TERMS = ["ADGM Companies Regulations", "ADGM Courts", "ADGM Registrar", "FSRA"]

# --- Helper utility ---
def _lower(text: Optional[str]) -> str:
    return (text or "").lower()

# --- Document-level checker ---
class DocumentComplianceChecker:
    def __init__(self, kb: ADGMKnowledgeBase):
        self.kb = kb
        self.rules = ADGMComplianceRules()

    def check_jurisdiction(self, content: DocumentContent) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        text = _lower(content.raw_text)
        required_found = any(re.search(p, text, flags=re.IGNORECASE) for p in self.rules.JURISDICTION_REQUIRED)
        forbidden_match = next((m.group(0) for p in self.rules.JURISDICTION_FORBIDDEN if (m := re.search(p, text, flags=re.IGNORECASE))), None)

        if not required_found:
            issues.append(ComplianceIssue(
                issue_id=f"jurisdiction_missing::{content.metadata.filename}",
                document_name=content.metadata.filename, section="Jurisdiction Clause",
                issue_type="missing_jurisdiction", description="No explicit ADGM jurisdiction/reference found.",
                severity=IssueSeverity.HIGH,
                adgm_reference=CITATION_DATABASE.get("missing_jurisdiction"),
                suggestion="Add an explicit clause referencing ADGM Courts.",
                confidence=0.88
            ))

        if forbidden_match:
            issues.append(ComplianceIssue(
                issue_id=f"jurisdiction_forbidden::{content.metadata.filename}",
                document_name=content.metadata.filename, section="Jurisdiction Clause",
                issue_type="incorrect_jurisdiction", description=f"Document references '{forbidden_match}' which is not ADGM jurisdiction.",
                severity=IssueSeverity.CRITICAL,
                adgm_reference=CITATION_DATABASE.get("incorrect_jurisdiction"),
                suggestion="Replace with wording specifying ADGM Courts.",
                confidence=0.95
            ))
        return issues

    def check_mandatory_clauses(self, content: DocumentContent) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        doc_type = content.metadata.document_type
        required_tokens = self.rules.MANDATORY_CLAUSES.get(doc_type, [])
        text = _lower(content.raw_text)
        
        for token in required_tokens:
            if token not in text and not all(w in text for w in token.split()):
                issue_key = f"missing_clause_{token}"
                issues.append(ComplianceIssue(
                    issue_id=f"missing_clause::{token.replace(' ', '_')}::{content.metadata.filename}",
                    document_name=content.metadata.filename, section="Mandatory Clauses",
                    issue_type="missing_mandatory_clause", description=f"Missing mandatory clause: {token}",
                    severity=IssueSeverity.MEDIUM,
                    adgm_reference=CITATION_DATABASE.get(issue_key, CITATION_DATABASE["default"]),
                    suggestion=f"Include a clause for '{token}' as per ADGM templates.",
                    confidence=0.78
                ))
        return issues

    def check_adgm_terms(self, content: DocumentContent) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        text = content.raw_text or ""
        if sum(1 for term in self.rules.ADGM_TERMS if term.lower() in text.lower()) < 1:
            issues.append(ComplianceIssue(
                issue_id=f"adgm_terms_low::{content.metadata.filename}",
                document_name=content.metadata.filename, section="Terminology",
                issue_type="insufficient_adgm_terms", description="Document lacks explicit ADGM-specific terminology.",
                severity=IssueSeverity.LOW,
                adgm_reference=CITATION_DATABASE.get("insufficient_adgm_terms"),
                suggestion="Add references to ADGM regulations/templates.",
                confidence=0.6
            ))
        return issues

    def check_formatting(self, content: DocumentContent) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        text = _lower(content.raw_text)
        if "signature" not in text and "signatory" not in text and "signed" not in text:
            issues.append(ComplianceIssue(
                issue_id=f"missing_signature::{content.metadata.filename}",
                document_name=content.metadata.filename, section="Execution / Signatures",
                issue_type="missing_signatures", description="No signature block or signatory language detected.",
                severity=IssueSeverity.MEDIUM,
                adgm_reference=CITATION_DATABASE.get("missing_signatures"),
                suggestion="Add signature blocks for required signatories.",
                confidence=0.7
            ))
        return issues

# --- Orchestrator ---
class ADGMComplianceValidator:
    PROCESS_KEYWORDS = {"company_incorporation": ["articles of association", "memorandum of association", "incorporation"], "licensing": ["license application", "licence"]}
    CHECKLISTS = {"company_incorporation": ["Articles of Association", "Memorandum of Association", "Board Resolution Template", "UBO Declaration Form", "Register of Members and Directors"], "licensing": ["License Application Form"]}

    def __init__(self, knowledge_base: ADGMKnowledgeBase):
        self.kb = knowledge_base
        self.doc_checker = DocumentComplianceChecker(self.kb)

    def detect_process(self, docs: List[DocumentContent]) -> str:
        scores = {k: 0 for k in self.PROCESS_KEYWORDS}
        for doc in docs:
            text_to_check = f"{doc.metadata.filename} {doc.raw_text or ''}".lower()
            for process, keywords in self.PROCESS_KEYWORDS.items():
                if any(kw in text_to_check for kw in keywords):
                    scores[process] += 1
        return max(scores, key=scores.get) if any(scores.values()) else "company_incorporation"

    def check_completeness(self, process_key: str, docs: List[DocumentContent]) -> Tuple[List[str], List[str]]:
        required = self.CHECKLISTS.get(process_key, [])
        present_files = [d.metadata.filename.lower() for d in docs]
        missing = [req for req in required if not any(req.lower().replace(" ", "") in name.replace(" ", "") for name in present_files)]
        return required, missing

    def validate_document(self, content: DocumentContent) -> ComplianceReport:
        all_issues = []
        all_issues.extend(self.doc_checker.check_jurisdiction(content))
        all_issues.extend(self.doc_checker.check_mandatory_clauses(content))
        all_issues.extend(self.doc_checker.check_adgm_terms(content))
        all_issues.extend(self.doc_checker.check_formatting(content))
        
        score = self._calc_score(all_issues)
        recommendations = self._generate_recommendations(all_issues, content.metadata.document_type)
        missing_clauses = [i.description.replace("Missing mandatory clause: ", "") for i in all_issues if i.issue_type == "missing_mandatory_clause"]
        summary = {"total_issues": len(all_issues), "critical_issues": len([i for i in all_issues if i.severity == IssueSeverity.CRITICAL]), "high_issues": len([i for i in all_issues if i.severity == IssueSeverity.HIGH]), "medium_issues": len([i for i in all_issues if i.severity == IssueSeverity.MEDIUM]), "low_issues": len([i for i in all_issues if i.severity == IssueSeverity.LOW])}
        
        return ComplianceReport(document_name=content.metadata.filename, document_type=content.metadata.document_type, overall_score=score, issues=all_issues, missing_clauses=missing_clauses, recommendations=recommendations, validation_summary=summary)

    def validate_all_documents(self, contents: List[DocumentContent]) -> Dict[str, Any]:
        process_key = self.detect_process(contents)
        required, missing = self.check_completeness(process_key, contents)
        all_issues_found = []
        per_doc_reports_data = []

        for doc in contents:
            report = self.validate_document(doc)
            
            
            report_dict = asdict(report)
            
          
            for issue_in_report in report_dict.get('issues', []):
                if isinstance(issue_in_report['severity'], Enum):
                     issue_in_report['severity'] = issue_in_report['severity'].value
            
            per_doc_reports_data.append(report_dict)

           
            for issue in report.issues:
                issue_dict = asdict(issue)
              
                issue_dict["severity"] = issue.severity.value
                all_issues_found.append(issue_dict)
        
        return {
            "process": process_key.replace("_", " ").title(),
            "documents_uploaded": len(contents),
            "required_documents": len(required),
            "missing_documents": missing,
            "issues_found": all_issues_found,
            "per_document_reports": per_doc_reports_data,
            "generated_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _calc_score(issues: List[ComplianceIssue]) -> float:
        weights = {IssueSeverity.CRITICAL: -25, IssueSeverity.HIGH: -15, IssueSeverity.MEDIUM: -8, IssueSeverity.LOW: -3}
        return round(max(100 + sum(weights.get(i.severity, 0) for i in issues), 0.0), 2)

    @staticmethod
    def _generate_recommendations(issues: List[ComplianceIssue], doc_type: str) -> List[str]:
        recs = []
        if any(i.severity == IssueSeverity.CRITICAL for i in issues): recs.append("Resolve critical issues immediately.")
        if any(i.issue_type == "incorrect_jurisdiction" for i in issues): recs.append("Update jurisdiction clauses to reference ADGM Courts.")
        if any(i.issue_type == "missing_mandatory_clause" for i in issues): recs.append("Add all mandatory clauses per ADGM templates.")
        if doc_type in ("articles_of_association", "memorandum_of_association"): recs.append("Compare document against official ADGM templates.")
        recs.append("Consider legal counsel review prior to submission.")
        return recs