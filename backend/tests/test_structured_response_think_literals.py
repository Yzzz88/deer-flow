"""Literal thinking tags in structured model output must remain data."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.gateway.routers import suggestions
from app.gateway.routers.suggestions import SuggestionMessage, SuggestionsRequest, _parse_json_string_list, generate_suggestions
from deerflow.runtime.goal import build_goal_state, evaluate_goal_completion, parse_goal_evaluation_response
from deerflow.utils import oneshot_llm


@pytest.mark.parametrize("prefix", ["", "<think>planning [example]</think>\n", "Note: <think>planning [example]</think>\n"])
@pytest.mark.parametrize("literal", ["<think>example</think>", "<think>", '<think note="a > b">中🙂</think>', r"\"<think>\""])
def test_suggestion_preserves_literal_think_tag(prefix, literal):
    import json

    response = prefix + json.dumps([f"Explain {literal}"])
    assert _parse_json_string_list(response) == [f"Explain {literal}"]


@pytest.mark.parametrize("prefix", ["", "<think>planning {example}</think>\n", "Note: <think>planning {example}</think>\n"])
@pytest.mark.parametrize("literal", ["<think>example</think>", "<think>", '<think note="a > b">中🙂</think>', r"\"<think>\""])
def test_goal_evaluation_preserves_literal_think_tag(prefix, literal):
    import json

    response = prefix + json.dumps({"satisfied": True, "reason": f"Found {literal}"})
    assert parse_goal_evaluation_response(response)["reason"] == f"Found {literal}"


def test_suggestion_preserves_literal_tag_with_trailing_reasoning():
    response = '["Explain <think>example</think>"]\n<think>check [example]</think>'
    assert _parse_json_string_list(response) == ["Explain <think>example</think>"]


def test_goal_evaluation_preserves_literal_tag_with_trailing_reasoning():
    response = '{"satisfied": true, "reason": "Found <think>example</think>"}\n<think>check {example}</think>'
    assert parse_goal_evaluation_response(response)["reason"] == "Found <think>example</think>"


def test_json_inside_inline_reasoning_is_not_accepted_as_an_answer():
    assert _parse_json_string_list('preamble <think>["not an answer"]</think>') is None
    with pytest.raises(ValueError):
        parse_goal_evaluation_response('preamble <think>{"satisfied": true}</think>')


@pytest.mark.asyncio
async def test_goal_evaluator_preserves_literal_tag_from_model_response():
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='<think>review</think>\n{"satisfied": true, "reason": "Found <think>example</think>"}')))
    result = await evaluate_goal_completion(
        build_goal_state("Check the example"),
        [HumanMessage(content="Check the example"), AIMessage(content="The example is present.")],
        model=model,
    )
    assert result["satisfied"] is True
    assert result["reason"] == "Found <think>example</think>"
    model.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_suggestions_route_preserves_literal_tag_from_model_response(monkeypatch):
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='<think>review</think>\n["Explain <think>example</think>"]')))
    monkeypatch.setattr(oneshot_llm, "create_chat_model", lambda **kwargs: model)
    monkeypatch.setattr(suggestions, "get_current_user_from_request", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(suggestions, "authorize_model_use", lambda *_args, **_kwargs: None)
    request = SuggestionsRequest(messages=[SuggestionMessage(role="user", content="Explain the tag"), SuggestionMessage(role="assistant", content="It marks reasoning.")], n=1, model_name=None)
    result = await generate_suggestions.__wrapped__("t1", request, request=None, config=SimpleNamespace(suggestions=SimpleNamespace(enabled=True)))
    assert result.suggestions == ["Explain <think>example</think>"]
    model.ainvoke.assert_awaited_once()
