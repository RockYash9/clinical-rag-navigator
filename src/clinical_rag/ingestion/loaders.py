"""Loaders for raw source documents (PDF, HTML, XML from PubMed/WHO/NICE/CDC).

TODO:
    - load_pdf(path) -> str
    - load_html(path) -> str
    - fetch_pubmed(query) -> list[dict]  (via E-utilities API)
Each loader should return raw text plus source metadata
(source_name, url, retrieved_date, tier: "general_guidelines" | "disease_specific").
"""
