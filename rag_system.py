 
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
from datetime import datetime

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import json

from .utils.vector_store import ADGMVectorStore
from .utils.adgm_rules import ADGMKnowledgeBase, ADGMDocumentType
from .components.document_parser import ADGMDocumentParser, DocumentClassificationResult

logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """Custom Ollama LLM wrapper for LangChain integration."""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call Ollama API to generate response."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for legal accuracy
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            return f"Error: Failed to generate response - {str(e)}"

class ADGMRAGSystem:
    """
    Advanced Retrieval-Augmented Generation system for ADGM legal document analysis.
    Combines vector search with LLM reasoning for accurate legal document processing.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        # Initialize components
        self.vector_store = ADGMVectorStore()
        self.knowledge_base = ADGMKnowledgeBase()
        self.document_parser = ADGMDocumentParser()
        self.llm = OllamaLLM(model_name=model_name)
        
        # Initialize system
        self.is_initialized = False
        self._initialize_system()
        
        # Analysis templates
        self.analysis_templates = self._load_analysis_templates()
        
    def _initialize_system(self) -> bool:
        """Initialize the RAG system with knowledge base."""
        try:
            logger.info("Initializing ADGM RAG system...")
            
            # Check if vector store is populated
            stats = self.vector_store.get_database_statistics()
            if stats['total_documents'] == 0:
                logger.info("Populating vector database with ADGM knowledge base...")
                if not self.vector_store.populate_knowledge_base():
                    logger.error("Failed to populate knowledge base")
                    return False
            
            self.is_initialized = True
            logger.info("ADGM RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"RAG system initialization failed: {str(e)}")
            return False
    
    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load prompt templates for different analysis tasks."""
        return {
            'document_classification': """
You are an expert ADGM legal document analyst. Analyze the following document and provide classification information.

Document Content:
{document_content}

Relevant ADGM Rules:
{relevant_rules}

Tasks:
1. Identify the document type based on ADGM standards
2. List key legal elements present
3. Identify any missing mandatory elements
4. Assess ADGM compliance status

Provide your analysis in a structured format with clear reasoning.
""",

            'red_flag_detection': """
You are an ADGM compliance expert. Review the following document for potential red flags and compliance issues.

Document Content:
{document_content}

Document Type: {document_type}

ADGM Compliance Requirements:
{compliance_requirements}

Known Red Flag Patterns:
{red_flag_patterns}

Tasks:
1. Identify any jurisdictional issues (should reference ADGM, not UAE Federal Courts)
2. Check for missing mandatory clauses
3. Detect ambiguous or non-binding language
4. Verify proper signature and witness sections
5. Ensure compliance with ADGM-specific requirements

For each issue found, provide:
- Specific location/section
- Severity level (Critical/High/Medium/Low)
- Clear explanation
- Suggested correction
- Relevant ADGM regulation reference

Format your response as a structured analysis with actionable recommendations.
""",

            'compliance_validation': """
You are an ADGM regulatory compliance specialist. Validate the following document against ADGM requirements.

Document Content:
{document_content}

Document Type: {document_type}

Applicable ADGM Rules:
{applicable_rules}

Document Checklist:
{document_checklist}

Tasks:
1. Verify all mandatory fields are present and complete
2. Confirm ADGM jurisdiction is properly specified
3. Check compliance with ADGM-specific formatting requirements
4. Validate legal language and binding commitments
5. Ensure proper references to ADGM regulations

Provide a comprehensive compliance report with:
- Overall compliance status
- Detailed findings for each requirement
- Priority recommendations for improvement
- References to specific ADGM regulations
""",

            'missing_documents': """
You are an ADGM incorporation specialist. Based on the uploaded documents, determine what additional documents may be required.

Uploaded Documents:
{uploaded_documents}

Process Type: {process_type}

ADGM Requirements for {process_type}:
{process_requirements}

Tasks:
1. Identify the apparent legal process (incorporation, licensing, etc.)
2. Compare uploaded documents against ADGM requirements
3. List any missing mandatory documents
4. Suggest optional documents that would strengthen the application
5. Provide guidance on next steps

Format your response as a clear checklist with explanations for each missing item.
"""
        }
    
    def analyze_document_comprehensive(
        self, 
        document_content: Dict[str, Any],
        classification_result: DocumentClassificationResult
    ) -> Dict[str, Any]:
        """
        Perform comprehensive document analysis using RAG system.
        
        Args:
            document_content: Extracted document content
            classification_result: Document classification results
            
        Returns:
            Comprehensive analysis results
        """
        if not self.is_initialized:
            return {"error": "RAG system not initialized"}
        
        try:
            analysis_results = {
                'document_name': document_content['file_name'],
                'document_type': classification_result.document_type.value,
                'analysis_date': datetime.now().isoformat(),
                'classification_confidence': classification_result.confidence_score,
                'issues_found': [],
                'compliance_status': 'Under Analysis',
                'recommendations': [],
                'adgm_references': [],
                'missing_elements': []
            }
            
            # Step 1: Get relevant ADGM rules
            relevant_rules = self._get_relevant_rules(
                classification_result.document_type.value,
                classification_result.identified_keywords
            )
            
            # Step 2: Perform red flag detection
            red_flags = self._detect_red_flags(
                document_content,
                classification_result,
                relevant_rules
            )
            analysis_results['issues_found'].extend(red_flags)
            
            # Step 3: Validate compliance
            compliance_issues = self._validate_compliance(
                document_content,
                classification_result,
                relevant_rules
            )
            analysis_results['issues_found'].extend(compliance_issues)
            
            # Step 4: Check for missing elements
            missing_elements = self._check_missing_elements(
                document_content,
                classification_result,
                relevant_rules
            )
            analysis_results['missing_elements'] = missing_elements
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(
                analysis_results['issues_found'],
                missing_elements,
                relevant_rules
            )
            analysis_results['recommendations'] = recommendations
            
            # Step 6: Determine overall compliance status
            analysis_results['compliance_status'] = self._assess_compliance_status(
                analysis_results['issues_found']
            )
            
            # Step 7: Extract ADGM references
            analysis_results['adgm_references'] = self._extract_adgm_references(
                relevant_rules
            )
            
            # Store processed document for future reference
            self.vector_store.add_processed_document(
                document_content['raw_text'],
                classification_result.document_type.value,
                analysis_results
            )
            
            logger.info(f"Comprehensive analysis completed for: {document_content['file_name']}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_relevant_rules(self, document_type: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Retrieve relevant ADGM rules using vector search."""
        try:
            # Search for relevant rules
            results = self.vector_store.get_relevant_rules(document_type, keywords)
            
            # Also get rules from knowledge base
            kb_rules = self.knowledge_base.search_relevant_rules(document_type, keywords)
            
            # Combine and format results
            relevant_rules = []
            for result in results[:5]:  # Top 5 vector search results
                relevant_rules.append({
                    'source': 'vector_search',
                    'content': result['document'],
                    'metadata': result['metadata'],
                    'relevance_score': 1 - result.get('distance', 0)
                })
            
            for rule in kb_rules[:3]:  # Top 3 knowledge base results
                relevant_rules.append({
                    'source': 'knowledge_base',
                    'rule_id': rule.rule_id,
                    'title': rule.title,
                    'description': rule.description,
                    'legal_authority': rule.legal_authority,
                    'compliance_requirements': rule.compliance_requirements
                })
            
            return relevant_rules
            
        except Exception as e:
            logger.error(f"Failed to get relevant rules: {str(e)}")
            return []
    
    def _detect_red_flags(
        self,
        document_content: Dict[str, Any],
        classification: DocumentClassificationResult,
        relevant_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect red flags using RAG-enhanced analysis."""
        try:
            # Get red flag patterns from vector store
            red_flag_results = self.vector_store.search_similar_content(
                query=f"red flags for {classification.document_type.value}",
                collection_name='compliance_patterns',
                top_k=5
            )
            
            # Prepare context for LLM
            red_flag_patterns = "\n".join([
                result['document'] for result in red_flag_results
            ])
            
            # Create prompt
            prompt = self.analysis_templates['red_flag_detection'].format(
                document_content=document_content['raw_text'][:3000],  # Limit content
                document_type=classification.document_type.value,
                compliance_requirements=self._format_compliance_requirements(relevant_rules),
                red_flag_patterns=red_flag_patterns
            )
            
            # Get LLM analysis
            llm_response = self.llm(prompt)
            
            # Parse LLM response into structured format
            red_flags = self._parse_red_flag_response(llm_response, document_content['file_name'])
            
            return red_flags
            
        except Exception as e:
            logger.error(f"Red flag detection failed: {str(e)}")
            return []
    
    def _validate_compliance(
        self,
        document_content: Dict[str, Any],
        classification: DocumentClassificationResult,
        relevant_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate document compliance using RAG analysis."""
        try:
            # Get compliance requirements
            compliance_results = self.vector_store.get_compliance_requirements(
                classification.document_type.value
            )
            
            # Get document checklist if available
            checklist_result = self.vector_store.get_document_checklist(
                self._map_document_to_process_type(classification.document_type)
            )
            
            # Prepare context
            applicable_rules = self._format_applicable_rules(relevant_rules)
            document_checklist = checklist_result['document'] if checklist_result else "No specific checklist found"
            
            # Create prompt
            prompt = self.analysis_templates['compliance_validation'].format(
                document_content=document_content['raw_text'][:3000],
                document_type=classification.document_type.value,
                applicable_rules=applicable_rules,
                document_checklist=document_checklist
            )
            
            # Get LLM analysis
            llm_response = self.llm(prompt)
            
            # Parse compliance issues
            compliance_issues = self._parse_compliance_response(llm_response, document_content['file_name'])
            
            return compliance_issues
            
        except Exception as e:
            logger.error(f"Compliance validation failed: {str(e)}")
            return []
    
    def _check_missing_elements(
        self,
        document_content: Dict[str, Any],
        classification: DocumentClassificationResult,
        relevant_rules: List[Dict[str, Any]]
    ) -> List[str]:
        """Check for missing mandatory elements."""
        try:
            missing_elements = []
            
            # Get mandatory fields from relevant rules
            for rule in relevant_rules:
                if rule.get('source') == 'knowledge_base':
                    for requirement in rule.get('compliance_requirements', []):
                        # Simple text search for mandatory elements
                        requirement_keywords = requirement.lower().split()
                        document_text = document_content['raw_text'].lower()
                        
                        if not any(keyword in document_text for keyword in requirement_keywords[:2]):
                            missing_elements.append(requirement)
            
            return missing_elements
            
        except Exception as e:
            logger.error(f"Missing elements check failed: {str(e)}")
            return []
    
    def _generate_recommendations(
        self,
        issues_found: List[Dict[str, Any]],
        missing_elements: List[str],
        relevant_rules: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Recommendations based on issues
        critical_issues = [issue for issue in issues_found if issue.get('severity') == 'critical']
        if critical_issues:
            recommendations.append("Address critical compliance issues immediately before proceeding")
        
        # Recommendations for missing elements
        if missing_elements:
            recommendations.append("Review and include all mandatory document elements")
            recommendations.extend([f"Add: {element}" for element in missing_elements[:3]])
        
        # General recommendations
        if any(issue.get('type') == 'jurisdiction_issue' for issue in issues_found):
            recommendations.append("Update all jurisdiction references to specify ADGM Courts")
        
        return recommendations
    
    def _assess_compliance_status(self, issues_found: List[Dict[str, Any]]) -> str:
        """Assess overall compliance status."""
        if not issues_found:
            return "Fully Compliant"
        
        critical_count = sum(1 for issue in issues_found if issue.get('severity') == 'critical')
        high_count = sum(1 for issue in issues_found if issue.get('severity') == 'high')
        
        if critical_count > 0:
            return "Non-Compliant (Critical Issues)"
        elif high_count > 2:
            return "Non-Compliant (Multiple High Priority Issues)"
        elif high_count > 0:
            return "Partially Compliant (High Priority Issues)"
        else:
            return "Mostly Compliant (Minor Issues)"
    
    def _extract_adgm_references(self, relevant_rules: List[Dict[str, Any]]) -> List[str]:
        """Extract ADGM legal references."""
        references = set()
        
        for rule in relevant_rules:
            if rule.get('source') == 'knowledge_base':
                references.add(rule.get('legal_authority', ''))
            elif rule.get('metadata', {}).get('legal_authority'):
                references.add(rule['metadata']['legal_authority'])
        
        return list(filter(None, references))
    
    def _parse_red_flag_response(self, llm_response: str, document_name: str) -> List[Dict[str, Any]]:
        """Parse LLM response for red flags into structured format."""
        red_flags = []
        
        # Simple parsing - in production, use more sophisticated parsing
        lines = llm_response.split('\n')
        current_issue = {}
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith(('issue:', 'red flag:', 'problem:')):
                if current_issue:
                    red_flags.append(current_issue)
                current_issue = {
                    'document': document_name,
                    'type': 'red_flag',
                    'description': line,
                    'severity': 'medium'
                }
            elif line.lower().startswith('severity:'):
                current_issue['severity'] = line.split(':', 1)[1].strip().lower()
            elif line.lower().startswith('suggestion:'):
                current_issue['suggestion'] = line.split(':', 1)[1].strip()
        
        if current_issue:
            red_flags.append(current_issue)
        
        return red_flags
    
    def _parse_compliance_response(self, llm_response: str, document_name: str) -> List[Dict[str, Any]]:
        """Parse LLM compliance response into structured format."""
        # Similar parsing logic as red flags
        return self._parse_red_flag_response(llm_response, document_name)
    
    def _format_compliance_requirements(self, relevant_rules: List[Dict[str, Any]]) -> str:
        """Format compliance requirements for prompt."""
        requirements = []
        for rule in relevant_rules[:3]:
            if rule.get('source') == 'knowledge_base':
                requirements.extend(rule.get('compliance_requirements', []))
        
        return '\n'.join(f"- {req}" for req in requirements[:10])
    
    def _format_applicable_rules(self, relevant_rules: List[Dict[str, Any]]) -> str:
        """Format applicable rules for prompt."""
        rules_text = []
        for rule in relevant_rules[:3]:
            if rule.get('source') == 'knowledge_base':
                rules_text.append(f"Rule: {rule.get('title', '')}")
                rules_text.append(f"Authority: {rule.get('legal_authority', '')}")
                rules_text.append(f"Description: {rule.get('description', '')}")
                rules_text.append("")
        
        return '\n'.join(rules_text)
    
    def _map_document_to_process_type(self, document_type: ADGMDocumentType) -> str:
        """Map document type to process type for checklist lookup."""
        mapping = {
            ADGMDocumentType.ARTICLES_OF_ASSOCIATION: 'company_incorporation',
            ADGMDocumentType.MEMORANDUM_OF_ASSOCIATION: 'company_incorporation',
            ADGMDocumentType.BOARD_RESOLUTION: 'company_incorporation',
            ADGMDocumentType.EMPLOYMENT_CONTRACT: 'employment_contract'
        }
        
        return mapping.get(document_type, 'general')
    
    def check_missing_documents(self, uploaded_files: List[str]) -> Dict[str, Any]:
        """Check for missing documents in uploaded set."""
        try:
            # Determine likely process type from uploaded files
            process_type = self._determine_process_type(uploaded_files)
            
            # Get requirements for this process
            process_requirements = self.knowledge_base.get_document_checklist(process_type)
            
            if not process_requirements:
                return {
                    'process_type': 'unknown',
                    'missing_documents': [],
                    'message': 'Could not determine document requirements'
                }
            
            # Compare uploaded files against requirements
            required_docs = process_requirements['required_documents']
            missing_docs = []
            
            for required_doc in required_docs:
                if not self._is_document_type_present(required_doc, uploaded_files):
                    missing_docs.append(required_doc)
            
            return {
                'process_type': process_requirements['process_name'],
                'total_required': len(required_docs),
                'uploaded_count': len(uploaded_files),
                'missing_count': len(missing_docs),
                'missing_documents': missing_docs,
                'message': self._generate_missing_docs_message(
                    process_requirements['process_name'],
                    len(uploaded_files),
                    len(required_docs),
                    missing_docs
                )
            }
            
        except Exception as e:
            logger.error(f"Missing documents check failed: {str(e)}")
            return {'error': f'Check failed: {str(e)}'}
    
    def _determine_process_type(self, uploaded_files: List[str]) -> str:
        """Determine most likely process type from uploaded file names."""
        incorporation_indicators = ['articles', 'memorandum', 'incorporation', 'resolution']
        employment_indicators = ['employment', 'contract', 'job']
        
        file_names_lower = ' '.join(uploaded_files).lower()
        
        incorporation_score = sum(1 for indicator in incorporation_indicators if indicator in file_names_lower)
        employment_score = sum(1 for indicator in employment_indicators if indicator in file_names_lower)
        
        if incorporation_score > employment_score:
            return 'company_incorporation'
        elif employment_score > 0:
            return 'employment_contract'
        else:
            return 'company_incorporation'  # Default
    
    def _is_document_type_present(self, required_doc: str, uploaded_files: List[str]) -> bool:
        """Check if a required document type is present in uploaded files."""
        required_keywords = required_doc.lower().split()
        
        for uploaded_file in uploaded_files:
            uploaded_lower = uploaded_file.lower()
            if any(keyword in uploaded_lower for keyword in required_keywords[:2]):
                return True
        
        return False
    
    def _generate_missing_docs_message(
        self, 
        process_name: str, 
        uploaded_count: int, 
        required_count: int, 
        missing_docs: List[str]
    ) -> str:
        """Generate user-friendly missing documents message."""
        if not missing_docs:
            return f"All required documents for {process_name} appear to be present."
        
        message = f"It appears that you're attempting {process_name}. "
        message += f"Based on ADGM requirements, you have uploaded {uploaded_count} out of {required_count} required documents. "
        
        if len(missing_docs) == 1:
            message += f"The missing document appears to be: '{missing_docs[0]}'."
        else:
            message += f"The missing documents appear to be: {', '.join(f\"'{doc}'\" for doc in missing_docs)}."
        
        return message