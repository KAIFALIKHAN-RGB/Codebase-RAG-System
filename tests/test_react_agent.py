from unittest.mock import MagicMock

import pytest

from src.agents.code_intelligence import (
    MAX_TOOL_CALLS,
    ToolBudgetExceeded,
    ToolValidationError,
)
from src.agents.react_agent import (
    AgentError,
    ReActAgent,
    _safe_sdk_error_metadata,
    generate_react_answer,
)


def _tools():
    tools = MagicMock()
    tools.repository = "repo"
    tools.tool_names = (
        "search_code",
        "lookup_symbol",
        "read_source",
        "related_files",
        "inspect_repository",
    )
    tools.tool_calls = 0
    return tools


def test_returns_immediate_final_answer_without_tools():
    llm = MagicMock(return_value="FINAL_ANSWER\nThe parser is in src/parser.py.")
    tools = _tools()

    result = ReActAgent("repo", tools=tools, llm=llm).run("Where is the parser?")

    assert result == "The parser is in src/parser.py."
    tools.call.assert_not_called()
    llm.assert_called_once()


def test_runs_one_tool_then_returns_final_answer():
    llm = MagicMock(
        side_effect=[
            'TOOL_CALL\n{"name":"search_code","arguments":{"query":"parser"}}',
            "FINAL_ANSWER\nThe parser is in src/parser.py.",
        ]
    )
    tools = _tools()
    tools.call.return_value = {"results": [{"metadata": {"file_path": "src/parser.py"}}]}

    result = ReActAgent("repo", tools=tools, llm=llm).run("Where is the parser?")

    assert result == "The parser is in src/parser.py."
    tools.call.assert_called_once_with("search_code", {"query": "parser"})
    assert llm.call_count == 2
    assert "TOOL_RESULT:" in llm.call_args_list[1].args[1]
    assert "src/parser.py" in llm.call_args_list[1].args[1]


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("search_code", {"query": "parser"}),
        ("lookup_symbol", {"symbol": "parse"}),
        ("read_source", {"file_path": "src/parser.py"}),
    ],
)
def test_routes_all_supported_tools(tool_name, arguments):
    llm = MagicMock(
        side_effect=[
            f"TOOL_CALL\n{{\"name\":\"{tool_name}\",\"arguments\":{arguments!r}}}".replace(
                "'", '"'
            ),
            "FINAL_ANSWER\nDone.",
        ]
    )
    tools = _tools()
    tools.call.return_value = {}

    assert ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.") == "Done."
    tools.call.assert_called_once_with(tool_name, arguments)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "result"),
    [
        (
            "related_files",
            {"file_path": "src/main.py"},
            {"source_file": "src/main.py", "related_files": ["src/helper.py"]},
        ),
        (
            "inspect_repository",
            {"limit": 2},
            {"files": ["src/main.py", "src/helper.py"]},
        ),
    ],
)
def test_routes_repository_inspection_tools_with_explicit_repository(
    tool_name,
    arguments,
    result,
):
    tools = _tools()
    tools.call.return_value = result
    llm = MagicMock(
        side_effect=[
            (
                f"TOOL_CALL\n"
                f'{{"name":"{tool_name}","arguments":{arguments!r}}}'.replace(
                    "'", '"'
                )
            ),
            "FINAL_ANSWER\nInspection complete.",
        ]
    )

    answer = ReActAgent("repo", tools=tools, llm=llm).run("Inspect the repository.")

    assert answer == "Inspection complete."
    tools.call.assert_called_once_with(tool_name, arguments)
    assert tools.repository == "repo"
    assert result
    assert "src/main.py" in llm.call_args_list[1].args[1]


def test_repository_inspection_tools_cannot_bypass_three_call_limit():
    tools = _tools()
    tools.call.side_effect = lambda name, arguments: (
        setattr(tools, "tool_calls", tools.tool_calls + 1) or {}
    )
    llm = MagicMock(
        return_value=(
            "TOOL_CALL\n"
            '{"name":"inspect_repository","arguments":{"limit":1}}'
        )
    )

    with pytest.raises(AgentError, match="limit"):
        ReActAgent("repo", tools=tools, llm=llm).run(
            "Inspect the repository repeatedly."
        )

    assert tools.call.call_count == MAX_TOOL_CALLS
    assert llm.call_count == MAX_TOOL_CALLS + 1


@pytest.mark.parametrize(
    "response",
    [
        'TOOL_CALL\n{"name":"missing","arguments":{}}',
        "TOOL_CALL\nnot-json",
        "Here is a tool call: search_code",
        'TOOL_CALL\n{"name":"search_code","arguments":{},"extra":true}',
    ],
)
def test_rejects_unknown_or_malformed_model_output(response):
    tools = _tools()
    llm = MagicMock(return_value=response)

    with pytest.raises(AgentError):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    tools.call.assert_not_called()


def test_logs_invalid_protocol_category_without_response_contents(caplog):
    tools = _tools()
    llm = MagicMock(return_value="ordinary model prose")

    with caplog.at_level("WARNING", logger="src.agents.react_agent"):
        with pytest.raises(AgentError, match="invalid protocol"):
            ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    assert "category=invalid_protocol" in caplog.text
    assert "ordinary model prose" not in caplog.text
    tools.call.assert_not_called()


def test_handles_invalid_tool_arguments_without_retrying():
    tools = _tools()
    tools.call.side_effect = ToolValidationError("invalid")
    llm = MagicMock(
        return_value='TOOL_CALL\n{"name":"search_code","arguments":{"query":""}}'
    )

    with pytest.raises(AgentError, match="rejected"):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    assert llm.call_count == 1


def test_handles_operational_tool_failure_without_extra_execution():
    tools = _tools()
    tools.call.side_effect = RuntimeError("internal detail")
    llm = MagicMock(
        return_value='TOOL_CALL\n{"name":"search_code","arguments":{"query":"x"}}'
    )

    with pytest.raises(AgentError, match="tool failed"):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    assert llm.call_count == 1
    assert tools.call.call_count == 1


def test_enforces_agent_tool_loop_limit():
    tools = _tools()
    tools.call.side_effect = lambda name, arguments: (
        setattr(tools, "tool_calls", tools.tool_calls + 1) or {}
    )
    llm = MagicMock(
        return_value='TOOL_CALL\n{"name":"search_code","arguments":{"query":"x"}}'
    )

    with pytest.raises(AgentError, match="limit"):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    assert tools.call.call_count == MAX_TOOL_CALLS
    assert llm.call_count == MAX_TOOL_CALLS + 1


def test_handles_tool_budget_exhaustion_without_extra_execution():
    tools = _tools()
    tools.call.side_effect = ToolBudgetExceeded("exhausted")
    llm = MagicMock(
        return_value='TOOL_CALL\n{"name":"search_code","arguments":{"query":"x"}}'
    )

    with pytest.raises(AgentError, match="limit"):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    assert llm.call_count == 1


def test_handles_empty_gemini_response():
    tools = _tools()
    llm = MagicMock(return_value="")

    with pytest.raises(AgentError, match="empty"):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    tools.call.assert_not_called()


def test_handles_gemini_exception_without_tool_execution():
    tools = _tools()
    llm = MagicMock(side_effect=TimeoutError("network detail"))

    with pytest.raises(AgentError, match="language model request failed"):
        ReActAgent("repo", tools=tools, llm=llm).run("Inspect code.")

    tools.call.assert_not_called()


def test_propagates_explicit_repository_to_default_tools():
    llm = MagicMock(return_value="FINAL_ANSWER\nDone.")
    tools = _tools()
    tools.repository = "repository-name"

    agent = ReActAgent("repository-name", tools=tools, llm=llm)

    assert agent.repository == "repository-name"


def test_rejects_tool_registry_for_different_repository():
    with pytest.raises(ValueError, match="does not match"):
        ReActAgent("repo-a", tools=_tools(), llm=MagicMock())


def test_gemini_adapter_sends_protocol_and_all_tool_schemas(monkeypatch):
    client = MagicMock()
    client.models.generate_content.return_value.text = (
        "FINAL_ANSWER\nThe answer is grounded."
    )
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )

    result = generate_react_answer("Where is parsing?", "TOOL_RESULT: {}")

    assert result == "FINAL_ANSWER\nThe answer is grounded."
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    for tool_name in (
        "search_code",
        "lookup_symbol",
        "read_source",
        "related_files",
        "inspect_repository",
    ):
        assert tool_name in prompt
    assert "FINAL_ANSWER" in prompt
    assert "TOOL_CALL" in prompt


def test_gemini_adapter_uses_bounded_timeout(monkeypatch):
    client = MagicMock()
    client.models.generate_content.return_value.text = "FINAL_ANSWER\nDone."
    monkeypatch.setattr("src.agents.react_agent._react_client", None)
    monkeypatch.setattr(
        "src.agents.react_agent.llm_client.api_key",
        "test-key",
    )
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("src.agents.react_agent.genai.Client", constructor)

    generate_react_answer("Question", "Context")

    options = constructor.call_args.kwargs["http_options"]
    assert options.timeout == 30_000


def test_gemini_adapter_handles_empty_and_sdk_failures(monkeypatch):
    client = MagicMock()
    client.models.generate_content.return_value.text = ""
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )

    with pytest.raises(AgentError, match="empty"):
        generate_react_answer("Question", "Context")

    failure = RuntimeError("secret detail")
    client.models.generate_content.side_effect = failure
    with pytest.raises(AgentError, match="Gemini request failed: RuntimeError") as error:
        generate_react_answer("Question", "Context")
    assert error.value.__cause__ is failure
    assert "secret detail" not in str(error.value)


def test_gemini_adapter_handles_missing_response_text(monkeypatch):
    client = MagicMock()
    client.models.generate_content.return_value = object()
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )

    with pytest.raises(AgentError, match="empty"):
        generate_react_answer("Question", "Context")


@pytest.mark.parametrize("status_code", [503, 504])
def test_gemini_adapter_retries_transient_status_once(status_code, monkeypatch):
    client = MagicMock()
    failure = RuntimeError("transient detail")
    failure.status_code = status_code
    client.models.generate_content.side_effect = [
        failure,
        MagicMock(text="FINAL_ANSWER\nRecovered."),
    ]
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )

    assert generate_react_answer("Question", "Context") == (
        "FINAL_ANSWER\nRecovered."
    )
    assert client.models.generate_content.call_count == 2


@pytest.mark.parametrize("status_code", [503, 504])
def test_gemini_adapter_exhausts_one_transient_retry(status_code, monkeypatch):
    client = MagicMock()
    first = RuntimeError("first transient detail")
    first.status_code = status_code
    second = RuntimeError("second transient detail")
    second.status_code = status_code
    client.models.generate_content.side_effect = [first, second]
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )

    with pytest.raises(AgentError, match="RuntimeError") as error:
        generate_react_answer("Question", "Context")

    assert error.value.__cause__ is second
    assert client.models.generate_content.call_count == 2


def test_gemini_adapter_does_not_retry_non_retryable_status(monkeypatch):
    client = MagicMock()
    failure = RuntimeError("non-retryable detail")
    failure.status_code = 400
    client.models.generate_content.side_effect = failure
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )

    with pytest.raises(AgentError, match="RuntimeError") as error:
        generate_react_answer("Question", "Context")

    assert error.value.__cause__ is failure
    assert client.models.generate_content.call_count == 1


def test_transient_retry_executes_tools_only_after_successful_response(monkeypatch):
    client = MagicMock()
    failure = RuntimeError("transient detail")
    failure.status_code = 503
    client.models.generate_content.side_effect = [
        failure,
        MagicMock(
            text='TOOL_CALL\n{"name":"search_code","arguments":{"query":"parser"}}'
        ),
        MagicMock(text="FINAL_ANSWER\nParser found."),
    ]
    monkeypatch.setattr(
        "src.agents.react_agent._get_react_client",
        lambda: client,
    )
    tools = _tools()
    tools.call.return_value = {}

    assert ReActAgent("repo", tools=tools, max_tool_calls=1).run("Find parser.") == (
        "Parser found."
    )

    assert client.models.generate_content.call_count == 3
    tools.call.assert_called_once_with(
        "search_code",
        {"query": "parser"},
    )


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"status_code": 503, "status": "Service Unavailable"}, (503, "Service Unavailable")),
        ({"code": "429", "reason": "Too Many Requests"}, ("429", "Too Many Requests")),
        ({"status_code": object(), "status": object()}, (None, None)),
    ],
)
def test_safe_sdk_error_metadata_extracts_only_safe_fields(attributes, expected):
    error = RuntimeError("secret request details")
    for name, value in attributes.items():
        setattr(error, name, value)

    assert _safe_sdk_error_metadata(error) == expected


def test_evaluation_direct_question_is_answered_without_tools():
    tools = _tools()
    llm = MagicMock(
        return_value=(
            "FINAL_ANSWER\n"
            "The repository uses FastAPI for its HTTP application."
        )
    )

    answer = ReActAgent(
        "repo",
        tools=tools,
        llm=llm,
    ).run("Which web framework does this repository use?")

    assert answer == "The repository uses FastAPI for its HTTP application."
    tools.call.assert_not_called()
    assert llm.call_count == 1


def test_evaluation_two_step_question_uses_first_result_for_second_tool():
    tools = _tools()
    tools.call.side_effect = [
        {
            "results": [
                {
                    "metadata": {"file_path": "src/parser/ast_parser.py"},
                    "code": "def parse_source(source): ...",
                }
            ]
        },
        {
            "repository": "repo",
            "file_path": "src/parser/ast_parser.py",
            "start_line": 1,
            "end_line": 3,
            "code": "def parse_source(source):\n    return ast.parse(source)\n",
        },
    ]

    def llm_response(question, context):
        if "TOOL_RESULT:" not in context:
            return (
                "TOOL_CALL\n"
                '{"name":"search_code","arguments":{"query":"source parser"}}'
            )
        if "ast.parse" not in context:
            assert "src/parser/ast_parser.py" in context
            return (
                "TOOL_CALL\n"
                '{"name":"read_source","arguments":'
                '{"file_path":"src/parser/ast_parser.py","start_line":1,"end_line":3}}'
            )
        assert "ast.parse" in context
        return (
            "FINAL_ANSWER\n"
            "The parser is implemented in "
            "src/parser/ast_parser.py and uses ast.parse."
        )

    answer_llm = MagicMock(side_effect=llm_response)

    answer = ReActAgent(
        "repo",
        tools=tools,
        llm=answer_llm,
    ).run("Where is source parsing implemented and how does it parse code?")

    assert answer == (
        "The parser is implemented in "
        "src/parser/ast_parser.py and uses ast.parse."
    )
    assert tools.call.call_args_list == [
        (
            ("search_code", {"query": "source parser"}),
            {},
        ),
        (
            (
                "read_source",
                {
                    "file_path": "src/parser/ast_parser.py",
                    "start_line": 1,
                    "end_line": 3,
                },
            ),
            {},
        ),
    ]
    assert "src/parser/ast_parser.py" in answer_llm.call_args_list[1].args[1]
    assert "ast.parse" in answer_llm.call_args_list[2].args[1]


def test_evaluation_insufficient_evidence_does_not_invent_code_facts():
    tools = _tools()
    tools.call.return_value = {"results": []}
    llm = MagicMock(
        side_effect=[
            "TOOL_CALL\n"
            '{"name":"search_code","arguments":{"query":"unknown feature"}}',
            "FINAL_ANSWER\n"
            "I cannot determine that from the available indexed code.",
        ]
    )

    answer = ReActAgent(
        "repo",
        tools=tools,
        llm=llm,
    ).run("What exact algorithm does the unknown feature use?")

    assert answer == "I cannot determine that from the available indexed code."
    tools.call.assert_called_once_with(
        "search_code",
        {"query": "unknown feature"},
    )
    assert "results" in llm.call_args_list[1].args[1]
    assert "unknown feature" not in answer
