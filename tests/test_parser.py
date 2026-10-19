import logging

from src.parser.ast_parser import parse_code


def test_parse_code_logs_syntax_errors_without_printing_source(
    caplog,
    capsys,
):
    with caplog.at_level(logging.WARNING, logger="src.parser.ast_parser"):
        result = parse_code("def broken(:\n    pass")

    assert result is None
    assert capsys.readouterr().out == ""
    assert (
        "Unable to parse Python source because of a syntax error."
        in caplog.messages
    )
