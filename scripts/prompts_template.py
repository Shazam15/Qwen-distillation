"""ATLAS's production RAG prompt template and system prompt, mirrored here so the
distillation set trains on the exact task shape ATLAS uses in production.

IMPORTANT: this is a COPY, not a live import - ATLAS (PDF-Assistant-RAG) is a
separate repository (see the plan's scope note). Before running b_build_prompts.py,
diff this file against backend/app/rag/prompts.py in your ATLAS checkout and paste
over RAG_PROMPT_TEMPLATE / SYSTEM_PROMPT below if they've drifted.

g_evaluate.py parses a rendered prompt back apart using the
"## Contexto del Documento" / "## Solicitud del Investigador" headers - keep those
two headers intact even if you reword the rest, or update g_evaluate.py to match.
"""

SYSTEM_PROMPT = (
    "Eres ATLAS, un asistente de investigacion academica. Respondes unicamente con "
    "base en los documentos proporcionados, citando cada afirmacion con su marcador "
    "[D1], [D2], etc. Si el contexto no contiene la informacion necesaria para "
    "responder, indicalo explicitamente en lugar de inventar una respuesta."
)

RAG_PROMPT_TEMPLATE = """## Contexto del Documento
{context}

## Solicitud del Investigador
{question}
{style_reference}"""
