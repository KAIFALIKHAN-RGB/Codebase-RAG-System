import ast
import logging


logger = logging.getLogger(__name__)

def parse_code(code):
    """
    Parses the given Python code and returns its Abstract Syntax Tree (AST).
    """
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError:
        logger.warning("Unable to parse Python source because of a syntax error.")
        return None