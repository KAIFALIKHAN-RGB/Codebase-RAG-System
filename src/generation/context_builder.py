def build_context(retrieved_chunks):
    """
    Convert retrieved code chunks into a structured context for the LLM.
    """

    if not retrieved_chunks:
        return ""

    context_parts = []

    source_number = 1
    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}
            
        code = chunk.get("code", "")

        if not isinstance(code, str) or not code.strip():
            continue

        file_path = metadata.get("file_path", "Unknown")
        name = metadata.get("name", "Unknown")
        start_line = metadata.get("start_line", "?")
        end_line = metadata.get("end_line", "?")

        context = (
            f"[Source {source_number}]\n"
            f"File: {file_path}\n"
            f"Symbol: {name}\n"
            f"Lines: {start_line} - {end_line}\n"
            f"Code:\n"
            f"```python\n"
            f"{code.strip()}\n"
            f"```"
        )

        context_parts.append(context)
        source_number += 1

    return "\n\n".join(context_parts)