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

    method_to_class = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef): 
            for child in node.body:
                # Extend parent-class tracking to async methods as well as regular methods.
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_to_class[id(child)] = node.name

    for node in ast.walk(tree):
        # Create chunks for regular functions, async functions, and classes.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):        
            parameters = []
            # Support parameter extraction for async functions just like regular functions.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [arg.arg for arg in node.args.args]

            docstring = ast.get_docstring(node)
            parent_class = method_to_class.get(id(node), None)
            start_line = node.lineno
            while (start_line > 1 and lines[start_line - 2].strip().startswith("@")):
                start_line -= 1

            return_type = None
            if hasattr(node, 'returns') and node.returns is not None:
                return_type = ast.unparse(node.returns)
            chunk = {
                "type" : type(node).__name__,
                "name" : node.name,
                "file_path" : str(file_path),
                "start_line" : start_line,
                "end_line" : node.end_lineno,
                "code" : "\n".join(lines[start_line - 1: node.end_lineno]),
                "parameters" : parameters,
                "docstring" : docstring,
                "parent_class" : parent_class,
                "return_type" : return_type
            }
            chunks.append(chunk)
    return chunks
