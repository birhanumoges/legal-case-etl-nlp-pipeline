"""rag/prompt_templates.py — Legal RAG prompt templates."""

LEGAL_QA_TEMPLATE = """You are an expert legal analyst. Use the following retrieved case excerpts to answer the question.

RETRIEVED CASES:
{context}

QUESTION: {question}

Provide a concise, accurate legal answer citing the relevant cases where applicable.
ANSWER:"""

CASE_SUMMARY_TEMPLATE = """Summarise the following legal case in 3-5 sentences, noting the case type, verdict, and key legal principles.

CASE TEXT:
{case_text}

SUMMARY:"""

VERDICT_EXPLANATION_TEMPLATE = """Given the following case facts and verdict, explain the legal reasoning.

CASE FACTS:
{facts}

VERDICT: {verdict}

LEGAL REASONING:"""

SIMILAR_CASES_TEMPLATE = """Based on the retrieved similar cases below, identify common legal patterns.

QUERY CASE: {query}

SIMILAR CASES:
{context}

PATTERNS:"""
