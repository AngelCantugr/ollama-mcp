def wrap_untrusted(model: str, response: str) -> str:
    """Wrap model output in untrusted envelope to prevent prompt injection.

    Local model output must never be treated as trusted Claude instructions.
    Escaping the closing tag prevents injection via crafted model responses.
    """
    escaped = response.replace("</ollama_output>", "<\\/ollama_output>")
    return f'<ollama_output model="{model}" untrusted="true">{escaped}</ollama_output>'
