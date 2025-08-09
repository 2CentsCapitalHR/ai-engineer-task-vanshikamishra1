from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
import re
from typing import List, Optional, Dict, Any

from .document_parser import DocumentContent
from .knowledge_base import ADGMKnowledgeBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------
# Models
# -------------------------
@dataclass
class KBReference:
    """A small container for KB search result metadata attached to flags."""
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "snippet": (self.snippet[:800] + "..." if self.snippet and len(self.snippet) > 800 else self.snippet),
            "source": self.source,
            "score": self.score
        }


@dataclass
class LegalFlag:
    """Represents a single red flag found in a document."""
    flag_id: str
    document_name: str
    section: str
    type: str
    description: str
    severity: str  # "critical" | "high" | "medium" | "low"
    confidence: float  # 0.0 - 1.0
    suggestion: Optional[str] = None
    kb_reference: Optional[KBReference] = None
    location_hint: Optional[str] = None  # e.g., "Clause 3.2" or "Paragraph starting '...'"


@dataclass
class LegalAnalysisReport:
    document_name: str
    document_type: str
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    flags: List[LegalFlag] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "document_type": self.document_type,
            "generated_at": self.generated_at,
            "flags": [self._flag_to_dict(f) for f in self.flags]
        }

    @staticmethod
    def _flag_to_dict(flag: LegalFlag) -> Dict[str, Any]:
        d = asdict(flag)
        if flag.kb_reference:
            d["kb_reference"] = flag.kb_reference.to_dict()
        return d


# -------------------------
# Analyzer
# -------------------------
class LegalAnalyzer:
    """
    LegalAnalyzer performs deterministic checks for common legal red flags then
    augments findings with ADGM KB references (when available).
    """

    # Patterns / heuristics
    AMBIGUOUS_PATTERNS = [
        r"\bmay\b(?!\s+be\b)",         # permissive
        r"\bshould\b(?!\s+not\b)",     # advisory
        r"\bendeavour\b|\bendeavor\b",
        r"\bbest\s+effort(s)?\b",
        r"\bcommercially\s+reasonable\b"
    ]

    NON_BINDING_PATTERNS = [
        r"\bsubject to\b",
        r"\bwithout prejudice\b",
        r"\bsubject to the discretion\b",
        r"\bin\s+principle\b"
    ]

    UNLIMITED_INDEMNITY_PATTERNS = [
        r"\bindemnif(y|ies|ication)\b",
        r"\bshall indemnify\b",
        r"\bindemnify and hold harmless\b"
    ]

    LIABILITY_CAP_PATTERN = r"\blimit(?:ed)?\s+to\s+([A-Z€£$0-9,.\s]+|[0-9]+%?)"

    GOVERNING_JURISDICTION_FORBIDDEN = [
        r"\bUAE Federal Court(s)?\b",
        r"\bDubai Courts?\b",
        r"\bAbu Dhabi Courts\b(?!.*Global)"
    ]

    UBO_MENTION_PATTERNS = [
        r"\bubo\b", r"\bultimate beneficial owner(s)?\b", r"\bbeneficial owner(s)?\b"
    ]

    SIGNATORY_PATTERNS = [
        r"\bsignature\b", r"\bsigned\b", r"\bsignatory\b", r"\bfor and on behalf of\b"
    ]

    ESCAPE_CLAUSE_PATTERNS = [
        r"\bforce majeure\b",
        r"\btermination for convenience\b",
        r"\bwithout liability\b",
    ]

    def __init__(self, knowledge_base: Optional[ADGMKnowledgeBase] = None):
        self.kb = knowledge_base

    # -------------------------
    # Public API
    # -------------------------
    def analyze_document(self, content: DocumentContent) -> LegalAnalysisReport:
        """
        Analyze one document and return a LegalAnalysisReport with identified flags.
        """
        logger.info("Analyzing document: %s", content.metadata.filename)
        report = LegalAnalysisReport(document_name=content.metadata.filename, document_type=content.metadata.document_type)

        text = (content.raw_text or "").strip()
        lower_text = text.lower()

        # Run detectors
        report.flags.extend(self._detect_ambiguous_language(content, lower_text))
        report.flags.extend(self._detect_non_binding_terms(content, lower_text))
        report.flags.extend(self._detect_indemnity_and_liability(content, lower_text))
        report.flags.extend(self._detect_governing_jurisdiction(content, lower_text))
        report.flags.extend(self._detect_missing_ubo(content, lower_text))
        report.flags.extend(self._detect_missing_signatory(content, lower_text))
        report.flags.extend(self._detect_escape_clauses(content, lower_text))

        # Add KB context to each flag where possible
        if self.kb:
            self._annotate_with_kb(report)

        logger.info("Analysis complete: %d flags found", len(report.flags))
        return report

    def analyze_documents(self, contents: List[DocumentContent]) -> List[LegalAnalysisReport]:
        """Run analyze_document for a list of DocumentContent items."""
        return [self.analyze_document(c) for c in contents]

    # -------------------------
    # Detectors (heuristics)
    # -------------------------
    def _detect_ambiguous_language(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        flags = []
        for pat in self.AMBIGUOUS_PATTERNS:
            for m in re.finditer(pat, lower_text, flags=re.IGNORECASE):
                snippet = self._snippet_around(lower_text, m.start(), m.end())
                flags.append(LegalFlag(
                    flag_id=f"ambiguous_{m.start()}",
                    document_name=content.metadata.filename,
                    section="General",
                    type="ambiguous_language",
                    description=f"Use of ambiguous/advisory wording: '{m.group(0)}' in context: \"{snippet}\"",
                    severity="medium",
                    confidence=0.7,
                    suggestion="Replace advisory wording with clear, binding obligations where appropriate.",
                    location_hint=snippet
                ))
        return flags

    def _detect_non_binding_terms(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        flags = []
        for pat in self.NON_BINDING_PATTERNS:
            for m in re.finditer(pat, lower_text, flags=re.IGNORECASE):
                snippet = self._snippet_around(lower_text, m.start(), m.end())
                flags.append(LegalFlag(
                    flag_id=f"nonbinding_{m.start()}",
                    document_name=content.metadata.filename,
                    section="General",
                    type="non_binding_term",
                    description=f"Non-binding or discretionary language found: '{m.group(0)}' in \"{snippet}\"",
                    severity="low",
                    confidence=0.6,
                    suggestion="If an obligation is intended, reword to use mandatory verbs (e.g., 'shall').",
                    location_hint=snippet
                ))
        return flags

    def _detect_indemnity_and_liability(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        flags = []
        # Detect indemnity mentions (potentially broad/unlimited)
        for m in re.finditer("|".join(self.UNLIMITED_INDEMNITY_PATTERNS), lower_text, flags=re.IGNORECASE):
            snippet = self._snippet_around(lower_text, m.start(), m.end())
            # try to find a liability cap near this match
            cap_match = re.search(self.LIABILITY_CAP_PATTERN, lower_text[m.end(): m.end() + 400], flags=re.IGNORECASE)
            if cap_match:
                cap = cap_match.group(1)
                severity = "medium"
                suggestion = f"Ensure indemnity is limited and proportionate; consider explicit cap such as {cap}."
            else:
                severity = "high"
                suggestion = "Consider adding a liability cap, carve-outs (e.g., fraud), and reasoned scope to the indemnity clause."
            flags.append(LegalFlag(
                flag_id=f"indemnity_{m.start()}",
                document_name=content.metadata.filename,
                section="Indemnity / Liability",
                type="indemnity_clause",
                description=f"Potentially broad indemnity language: '{m.group(0)}' in \"{snippet}\"",
                severity=severity,
                confidence=0.78 if cap_match else 0.9,
                suggestion=suggestion,
                location_hint=snippet
            ))
        # Check for very broadly worded limitation / no-cap escape
        # If 'limit to' phrase present, we consider it positive; otherwise a missing cap is a flag
        if re.search(r"\bindemnif(y|ies|ication)\b", lower_text, flags=re.IGNORECASE) and not re.search(self.LIABILITY_CAP_PATTERN, lower_text, flags=re.IGNORECASE):
            # If indemnity exists but no cap is found anywhere
            flags.append(LegalFlag(
                flag_id="indemnity_nocap",
                document_name=content.metadata.filename,
                section="Indemnity / Liability",
                type="missing_liability_cap",
                description="Indemnity language detected but no clear monetary cap or limitation found in the surrounding text.",
                severity="high",
                confidence=0.9,
                suggestion="Add a clear monetary cap or proportionate limitation to indemnities, or explain why uncapped indemnity is justified.",
                location_hint=None
            ))
        return flags

    def _detect_governing_jurisdiction(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        flags = []
        # Check for forbidden jurisdictions
        for pat in self.GOVERNING_JURISDICTION_FORBIDDEN:
            m = re.search(pat, lower_text, flags=re.IGNORECASE)
            if m:
                snippet = self._snippet_around(lower_text, m.start(), m.end())
                flags.append(LegalFlag(
                    flag_id=f"jurisdiction_conflict_{m.start()}",
                    document_name=content.metadata.filename,
                    section="Governing Law / Jurisdiction",
                    type="governing_jurisdiction_conflict",
                    description=f"References non-ADGM jurisdiction '{m.group(0)}' which may be incompatible with ADGM filing.",
                    severity="critical",
                    confidence=0.95,
                    suggestion="Replace governing law / jurisdiction wording with ADGM Courts or otherwise align with ADGM requirements.",
                    location_hint=snippet
                ))
        return flags

    def _detect_missing_ubo(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        """For incorporation-related docs, check for UBO mentions; if absent, flag recommend UBO disclosure"""
        flags = []
        # Only flag if the document type relates to incorporation
        doc_type = (content.metadata.document_type or "").lower()
        if "incorporation" in doc_type or "articles" in doc_type or "memorandum" in doc_type:
            if not any(re.search(pat, lower_text, flags=re.IGNORECASE) for pat in self.UBO_MENTION_PATTERNS):
                flags.append(LegalFlag(
                    flag_id="ubo_missing",
                    document_name=content.metadata.filename,
                    section="UBO / Beneficial Ownership",
                    type="missing_ubo_declaration",
                    description="No reference to Ultimate Beneficial Owner (UBO) or related disclosures found in incorporation paperwork.",
                    severity="high",
                    confidence=0.85,
                    suggestion="Include UBO declaration as per ADGM registration requirements (UBO details, % ownership, nationality).",
                    location_hint=None
                ))
        return flags

    def _detect_missing_signatory(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        flags = []
        # If document appears to be a formal filing but no signature tokens present
        doc_type = (content.metadata.document_type or "").lower()
        if doc_type in ("articles_of_association", "memorandum_of_association", "incorporation_application", "board_resolution"):
            if not any(re.search(pat, lower_text, flags=re.IGNORECASE) for pat in self.SIGNATORY_PATTERNS):
                flags.append(LegalFlag(
                    flag_id="signatory_missing",
                    document_name=content.metadata.filename,
                    section="Execution / Signatures",
                    type="missing_signatory_block",
                    description="Document appears to be a formal filing but no clear signature block or signatory language detected.",
                    severity="medium",
                    confidence=0.8,
                    suggestion="Add signature blocks including printed names, titles, and dates for execution.",
                    location_hint=None
                ))
        return flags

    def _detect_escape_clauses(self, content: DocumentContent, lower_text: str) -> List[LegalFlag]:
        flags = []
        for pat in self.ESCAPE_CLAUSE_PATTERNS:
            for m in re.finditer(pat, lower_text, flags=re.IGNORECASE):
                snippet = self._snippet_around(lower_text, m.start(), m.end())
                severity = "low"
                suggestion = "Ensure the clause is balanced and includes appropriate notice, mitigation obligations and time-limited relief where appropriate."
                if "termination for convenience" in m.group(0).lower():
                    severity = "medium"
                    suggestion = "Termination for convenience clauses should include notice periods and any compensation mechanics."
                flags.append(LegalFlag(
                    flag_id=f"escape_{m.start()}",
                    document_name=content.metadata.filename,
                    section="Remedies / Termination",
                    type="escape_clause",
                    description=f"Escape/termination style clause found: '{m.group(0)}' in \"{snippet}\"",
                    severity=severity,
                    confidence=0.65,
                    suggestion=suggestion,
                    location_hint=snippet
                ))
        return flags

    # -------------------------
    # KB annotation
    # -------------------------
    def _annotate_with_kb(self, report: LegalAnalysisReport) -> None:
        """
        For each flag, run a small KB search to find supporting ADGM guidance/snippets and attach
        the top result as a KBReference (if KB is available).
        """
        if not self.kb:
            return

        for flag in report.flags:
            try:
                query = self._compose_kb_query(flag)
                if not query:
                    continue
                results = self.kb.search_knowledge(query, n_results=2) or {}
                # results expected as {'documents': [[snippet1, snippet2,...], ...], 'meta': ...}
                docs = results.get("documents") or []
                snippet = None
                source = None
                score = None
                title = None
                # pick first usable snippet
                if isinstance(docs, list) and docs:
                    # docs may be list of lists (groups); flatten first group
                    first_group = docs[0] if isinstance(docs[0], list) else docs
                    if first_group:
                        snippet = first_group[0] if isinstance(first_group, list) else first_group
                        # attempt to extract source/score if returned by KB (best-effort)
                        meta = results.get("meta", {})
                        source = meta.get("source") if isinstance(meta, dict) else None
                if snippet:
                    flag.kb_reference = KBReference(
                        title=title,
                        snippet=snippet if isinstance(snippet, str) else str(snippet),
                        source=source,
                        score=score
                    )
            except Exception as e:
                logger.exception("KB annotation failed for flag %s: %s", flag.flag_id, e)

    @staticmethod
    def _compose_kb_query(flag: LegalFlag) -> Optional[str]:
        """Compose a conservative query string for the KB based on flag type."""
        if flag.type == "governing_jurisdiction_conflict":
            return "ADGM jurisdiction governing law courts replace UAE Federal Courts ADGM guidance"
        if flag.type == "missing_ubo_declaration":
            return "ADGM UBO declaration beneficial owner requirements registration checklist"
        if flag.type == "indemnity_clause" or flag.type == "missing_liability_cap":
            return "ADGM indemnity liability cap guidance contract indemnity ADGM"
        if flag.type == "ambiguous_language" or flag.type == "non_binding_term":
            return "ADGM drafting guidance mandatory wording 'shall' vs 'may' templates"
        if flag.type == "missing_signatory_block":
            return "ADGM document execution signature block requirements"
        if flag.type == "escape_clause":
            return "ADGM force majeure termination for convenience guidance"
        # fallback: search the doc type + flag phrase
        return f"ADGM {flag.type} guidance"

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _snippet_around(text: str, start: int, end: int, width: int = 80) -> str:
        """Return a short snippet around a match for context (usable as location_hint)."""
        s = max(0, start - width)
        e = min(len(text), end + width)
        snippet = text[s:e].strip()
        # clean up whitespace for readability
        snippet = re.sub(r"\s+", " ", snippet)
        # upcase first letter for readability in UI
        return snippet[:1000]

