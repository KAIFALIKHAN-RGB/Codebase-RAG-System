import json
import logging
import time
from typing import Any, Callable

from google import genai
from google.genai import types

from src.agents.code_intelligence import (
    MAX_TOOL_CALLS,
    CodeIntelligenceTools,
    ToolBudgetExceeded,
    ToolValidationError,
)
from src.generation import llm_client


class AgentError(RuntimeError):
    """Raised when the bounded agent cannot complete a request."""


LLMCallable = Callable[[str, str], str]
REACT_REQUEST_TIMEOUT_MS = 30_000
_react_client = None
logger = logging.getLogger(__name__)


def _safe_sdk_error_metadata(error: BaseException) -> tuple[Any, str | None]:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, (int, str)) or isinstance(status_code, bool):
        status_code = None

    status_text = getattr(error, "status", None)
    if status_text is None:
        status_text = getattr(error, "reason", None)
    if status_text is None and response is not None:
        status_text = getattr(response, "reason", None)
    if not isinstance(status_text, str):
        status_text = None
    else:
        status_text = " ".join(status_text.split())[:80] or None

    return status_code, status_text


def _get_react_client():
    global _react_client

    if _react_client is None:
        if not llm_client.api_key:
            raise AgentError("GEMINI_API_KEY is not set in the environment.")
        _react_client = genai.Client(
            api_key=llm_client.api_key,
            http_options=types.HttpOptions(timeout=REACT_REQUEST_TIMEOUT_MS),
        )
    return _react_client


def generate_react_answer(question: str, context: str) -> str:
    """Generate one bounded ReAct response using the existing Gemini setup."""
    prompt = f"""You are a bounded codebase investigation agent.
Use only the supplied repository context and tool results. Do not invent code facts.
Return exactly one protocol form:

{ReActAgent.PROTOCOL}

Tool schemas:
- search_code: {{"query": string, "k": optional positive integer <= 5, "threshold": optional number}}
- lookup_symbol: {{"symbol": string, "symbol_type": optional string}}
- read_source: {{"file_path": repository-relative string, "start_line": optional positive integer, "end_line": optional positive integer, "max_characters": optional integer <= 12000}}
- related_files: {{"file_path": repository-relative Python file path}}
- inspect_repository: {{"limit": optional positive integer <= 20}}

Repository question:
{question}

Current agent transcript:
{context}
"""
    for attempt in range(2):
        started_at = time.monotonic()
        try:
            response = _get_react_client().models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
            )
        except Exception as error:
            status_code, status_text = _safe_sdk_error_metadata(error)
            logger.warning(
                "Gemini request failed: exception_type=%s status_code=%s "
                "status_text=%s",
                type(error).__name__,
                status_code,
                status_text,
            )
            if attempt == 0 and status_code in (503, 504, "503", "504"):
                logger.info("Gemini request retry: status_code=%s", status_code)
                continue
            raise AgentError(
                f"Gemini request failed: {type(error).__name__}"
            ) from error
        finally:
            logger.info(
                "Gemini model call: elapsed_ms=%d prompt_length=%d",
                round((time.monotonic() - started_at) * 1000),
                len(prompt),
            )
        break

    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise AgentError("Gemini returned an empty response.")
    return response_text


class ReActAgent:
    """Bounded manual ReAct controller using the existing LLM boundary."""

    PROTOCOL = """Respond with exactly one of these forms:
FINAL_ANSWER
<answer text>

or:
TOOL_CALL
{"name":"<tool name>","arguments":{...}}

Allowed tools: search_code, lookup_symbol, read_source, related_files, inspect_repository.
Do not use Markdown fences or add text before or after the selected form.
"""

    def __init__(
        self,
        repository: str,
        *,
        tools: CodeIntelligenceTools | None = None,
        llm: LLMCallable | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ):
        if not isinstance(max_tool_calls, int) or isinstance(max_tool_calls, bool):
            raise ValueError("max_tool_calls must be an integer.")
        if max_tool_calls <= 0 or max_tool_calls > MAX_TOOL_CALLS:
            raise ValueError(
                f"max_tool_calls must be between 1 and {MAX_TOOL_CALLS}."
            )

        self.repository = repository
        if tools is not None and tools.repository != repository:
            raise ValueError("Tool registry repository does not match the agent repository.")
        self.tools = tools or CodeIntelligenceTools(repository)
        self.llm = generate_react_answer if llm is None else llm
        self.max_tool_calls = max_tool_calls

    def run(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise AgentError("Question must be a non-empty string.")

        transcript = self.PROTOCOL
        tool_calls = 0
        for _ in range(self.max_tool_calls + 1):
            response = self._generate(question, transcript)
            action, payload = self._parse_response(response)

            if action == "final":
                return payload

            if tool_calls >= self.max_tool_calls:
                raise AgentError("The agent tool-call limit was reached.")

            tool_name, arguments = payload
            tool_calls += 1
            try:
                result = self.tools.call(tool_name, arguments)
            except ToolBudgetExceeded:
                raise AgentError("The agent tool-call limit was reached.") from None
            except ToolValidationError:
                raise AgentError("The tool request was rejected.") from None
            except Exception:
                raise AgentError("The tool failed while processing the request.") from None

            tool_call = json.dumps(
                {"name": tool_name, "arguments": arguments},
                sort_keys=True,
            )
            tool_result = json.dumps(result, sort_keys=True, default=str)
            transcript = (
                f"{transcript}\nUSER QUESTION:\n{question}\n"
                f"TOOL_CALL:\n{tool_call}\n"
                f"TOOL_RESULT:\n{tool_result}\n"
                "Continue with exactly one protocol form."
            )

        raise AgentError("The agent tool-call limit was reached.")

    def _generate(self, question: str, context: str) -> str:
        try:
            response = self.llm(question, context)
        except AgentError:
            raise
        except Exception:
            raise AgentError("The language model request failed.") from None

        if not isinstance(response, str) or not response.strip():
            logger.warning("ReAct response rejected: category=empty_response")
            raise AgentError("The language model returned an empty response.")
        return response.strip()

    def _parse_response(self, response: str) -> tuple[str, Any]:
        if response.startswith("FINAL_ANSWER\n"):
            answer = response[len("FINAL_ANSWER\n") :].strip()
            if not answer:
                logger.warning("ReAct response rejected: category=empty_final_answer")
                raise AgentError("The final answer was empty.")
            return "final", answer

        if not response.startswith("TOOL_CALL\n"):
            logger.warning("ReAct response rejected: category=invalid_protocol")
            raise AgentError("The language model returned an invalid protocol response.")

        raw_call = response[len("TOOL_CALL\n") :]
        try:
            request = json.loads(raw_call)
        except (json.JSONDecodeError, TypeError):
            logger.warning("ReAct response rejected: category=malformed_json")
            raise AgentError("The tool-call JSON was invalid.") from None

        if (
            not isinstance(request, dict)
            or set(request) != {"name", "arguments"}
            or not isinstance(request["name"], str)
            or not isinstance(request["arguments"], dict)
        ):
            logger.warning("ReAct response rejected: category=invalid_tool_call_shape")
            raise AgentError("The tool-call JSON shape was invalid.")

        if request["name"] not in self.tools.tool_names:
            logger.warning("ReAct response rejected: category=unknown_tool")
            raise AgentError("The requested tool is not allowed.")

        return "tool", (request["name"], request["arguments"])
