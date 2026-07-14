def build_context(retrieved_chunks):
    """
    Convert relevant code chunks into a structured context for the LLM.
    """
    if not retrieved_chunks:
        return ""
    
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        code = chunk.get("code", "")

        file_path = metadata.get("file_path", "Unknown")
        name = metadata.get("name", "Unknown")
        start_line = metadata.get("start_line", "?")
        end_line = metadata.get("end_line", "?")

        context = (
            f"[Source {i}]\n"
            f"File: {file_path}\n"
            f"Symbol: {name}\n"
            f"Lines: {start_line} - {end_line}\n"
            f"Code:\n{code}"
        )
        context_parts.append(context)
    return "\n\n".join(context_parts)
