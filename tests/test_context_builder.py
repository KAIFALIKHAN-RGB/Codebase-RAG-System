from src.generation.context_builder import build_context


def test_build_context_with_valid_chunk():
    chunks = [
        {
            "code": "def hello():\n    return 'world'",
            "metadata": {
                "file_path": "src/main.py",
                "name": "hello",
                "start_line": 1,
                "end_line": 2,
            },
        }
    ]

    context = build_context(chunks)

    assert "[Source 1]" in context
    assert "File: src/main.py" in context
    assert "Symbol: hello" in context
    assert "Lines: 1 - 2" in context
    assert "def hello():" in context
    assert "```python" in context


def test_build_context_empty_chunks():
    assert build_context([]) == ""


def test_build_context_skips_invalid_chunks():
    chunks = [
        None,
        "invalid",
        {"code": ""},
        {"code": "   "},
        {
            "code": "def valid():\n    pass",
            "metadata": {
                "file_path": "valid.py",
                "name": "valid",
            },
        },
    ]

    context = build_context(chunks)

    assert "[Source 5]" in context
    assert "def valid():" in context
    assert "invalid" not in context


def test_build_context_handles_missing_metadata():
    chunks = [
        {
            "code": "x = 10",
            "metadata": {},
        }
    ]

    context = build_context(chunks)

    assert "File: Unknown" in context
    assert "Symbol: Unknown" in context
    assert "Lines: ? - ?" in context
    assert "x = 10" in context

def test_build_context_with_missing_metadata():
    chunks = [
        {
            "code": "def hello():\n    return 'world'",
            "metadata": {}
        }
    ]

    context = build_context(chunks)

    assert "[Source 1]" in context
    assert "File: Unknown" in context
    assert "Symbol: Unknown" in context
    assert "Lines: ? - ?" in context
    assert "def hello():" in context


def test_build_context_skips_invalid_chunks():
    chunks = [
        None,
        "invalid",
        {
            "code": "",
            "metadata": {}
        },
        {
            "code": "def valid():\n    return True",
            "metadata": {
                "file_path": "src/main.py",
                "name": "valid",
                "start_line": 1,
                "end_line": 2
            }
        }
    ]

    context = build_context(chunks)

    assert "[Source 1]" in context
    assert "def valid():" in context
    assert "invalid" not in context


def test_build_context_with_multiple_chunks():
    chunks = [
        {
            "code": "def first():\n    pass",
            "metadata": {
                "file_path": "src/first.py",
                "name": "first",
                "start_line": 1,
                "end_line": 2
            }
        },
        {
            "code": "def second():\n    pass",
            "metadata": {
                "file_path": "src/second.py",
                "name": "second",
                "start_line": 5,
                "end_line": 6
            }
        }
    ]

    context = build_context(chunks)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "src/first.py" in context
    assert "src/second.py" in context
    assert "def first():" in context
    assert "def second():" in context


def test_build_context_empty_chunks():
    assert build_context([]) == ""
    assert build_context(None) == ""

def test_build_context_handles_invalid_metadata_type():
    chunks = [
        {
            "code": "x = 10",
            "metadata": "invalid metadata"
        }
    ]

    context = build_context(chunks)

    assert "File: Unknown" in context
    assert "Symbol: Unknown" in context
    assert "Lines: ? - ?" in context
    assert "x = 10" in context