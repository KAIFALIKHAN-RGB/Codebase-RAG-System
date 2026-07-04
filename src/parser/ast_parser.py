import ast

def parse_code(code):
    """
    Parses the given Python code and returns its Abstract Syntax Tree (AST).
    """
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError as e:
        print(f"Syntax error while parsing code: {e}")
        return None