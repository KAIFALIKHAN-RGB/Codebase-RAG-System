from pathlib import Path

def read_python_file(file_path):
  """Reads the content of a Python file and returns it as a string."""

  with open(file_path, 'r', encoding='utf-8') as file:
    code = file.read()

  return code