import ast

def extract_chunks(tree, code, file_path):
    """
    Extracts chunks of code from the AST tree and returns them as a list of dictionaries.

    Args:
        tree (ast.AST): The AST tree of the code.
        code (str): The original code as a string.
        file_path (str): The path to the file being processed.
        """
    
    chunks = []
    lines = code.splitlines()

    class_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef): 
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    class_map[child.name] = node.name

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):        
            parameters = []
            if isinstance(node, ast.FunctionDef):
                parameters = [arg.arg for arg in node.args.args]

            docstring = ast.get_docstring(node)
            parent_class = class_map.get(node.name, None)
            chunk = {
                "type" : type(node).__name__,
                "name" : node.name,
                "file_path" : str(file_path),
                "start_line" : node.lineno,
                "end_line" : node.end_lineno,
                "code" : "\n".join(lines[node.lineno - 1: node.end_lineno]),
                "parameters" : parameters,
                "docstring" : docstring,
                "parent_class" : parent_class
            }
            chunks.append(chunk)
    return chunks
