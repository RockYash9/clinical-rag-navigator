"""System/user prompt templates enforcing context-grounded, citation-ready answers.

TODO:
    - build_prompt(question, retrieved_chunks) -> str
Prompt must instruct the model to answer ONLY from provided context and to say
"the corpus does not cover this" when retrieval is insufficient.
"""
