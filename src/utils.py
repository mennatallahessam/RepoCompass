"""
Common LLM initialization and resilient JSON extraction utilities for RepoCompass.
"""
import json
import os
import re
from typing import Any, Dict
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default gateway credentials fallback if not provided via environment variables
_DEFAULT_API_KEY = (
    "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJhZXNKN2kxNGNidnVuTU40MTJrOU5yZ2ROeENhTlJudTNPbC1TU08ycFlJIn0."
    "eyJleHAiOjE3NzA0NDI4MTIsImlhdCI6MTc3MDQ0MTAxMiwiYXV0aF90aW1lIjoxNzcwNDQxMDExLCJqdGkiOiJmZjUxMTdhNC04NGE2LTRjMjkt"
    "Yjk0OC0wMTg3NzY2MWZkODciLCJpc3MiOiJodHRwczovL2F1dGgubWNraW5zZXkuaWQvYXV0aC9yZWFsbXMvciIsImF1ZCI6ImJjZDIzNzI4LTNk"
    "MjctNDQ3Yy1hMGE5LWVhY2FmMzkzYTZmNSIsInN1YiI6IjZhZjEyNDMxLWUzYWMtNGM3Mi1hYjZhLTY5ZjI4ODlmYzZmYSIsInR5cCI6IklEIiwi"
    "YXpwIjoiYmNkMjM3MjgtM2QyNy00NDdjLWEwYTktZWFjYWYzOTNhNmY1Iiwic2Vzc2lvbl9zdGF0ZSI6ImRiNzUwODQ5LWJlMmUtNDk4NC04NmZk"
    "LWMyOGY3MThjZWFhZiIsImF0X2hhc2giOiIzMlVsT3dMMm8wUlNvUmVtX3gzUDlRIiwibmFtZSI6IlJpY2hhcmQgQ291cGVydGh3YWl0ZSIsImdp"
    "dmVuX25hbWUiOiJSaWNoYXJkIiwiZmFtaWx5X25hbWUiOiJDb3VwZXJ0aHdhaXRlIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYjJiZWVkNDk3Yjdk"
    "YzRmZiIsImVtYWlsIjoiUmljaGFyZF9Db3VwZXJ0aHdhaXRlQG1ja2luc2V5LmNvbSIsImFjciI6IjEiLCJzaWQiOiJkYjc1MDg0OS1iZTJlLTQ5"
    "ODQtODZmZC1jMjhmNzE4Y2VhYWYiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZm1ubyI6IjMxNjczNyIsImdyb3VwcyI6WyIyOTA3ZmExYy02ODNm"
    "LTRjZjctODU1NS04YzM0YTE2Yzg4ZDgiLCJBbGwgRmlybSBVc2VycyIsIjAzYjE1YTM0LTY1NWUtNGVjNC04MzRhLWI5ZGRhMTk3MjBmMCJdfQ."
    "ZSSU7ZL8l5o3KFoU0fhBtwapRddVm5Nz9o1mnIBj_3XYifi77shBUsRVm2cVEL3uxLTwYpybBSet2Q64O4NZJYOlfjjlukZHUeiot79mL3iTDyprz"
    "ns34ldBnWtIbUZD4S-VF7FCPbgsRmQIlPlm8BCR9UQhjgQ3Lt_fCOuyfQli5-RZU6ypESIzFsZ-X3ROKFzzmym1fif0paxcyfZiZx8gFrI_Py0fx"
    "a1pznuiSNGzJRcnSmuKQIARn0AfJGvm9VB07e5btuslRx4w1sexSy9vbJayPezt0EL3_-RDDsAUZ7u3Q3PjgVlcRfv0INgPpQjBncmSFvjsB-IPxkep8g"
)
_DEFAULT_BASE_URL = "https://openai.prod.ai-gateway.quantumblack.com/2907fa1c-683f-4cf7-8555-8c34a16c88d8/v1"


def configure_llm_env() -> None:
    """Ensure OpenAI environment variables are populated."""
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _DEFAULT_API_KEY
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = _DEFAULT_BASE_URL


def create_chat_model(temperature: float = 0.1, model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    configure_llm_env()
    return ChatOpenAI(model=model_name, temperature=temperature)


def invoke_llm_text(llm: ChatOpenAI, prompt: str) -> str:
    """Invoke the language model and return its string content."""
    reply = llm.invoke(prompt)
    return reply.content if hasattr(reply, "content") else str(reply)


def _normalize_array_key_value_pairs(raw_json: str) -> str:
    """
    Sanitize malformed JSON arrays where an LLM accidentally emits
    `"pkg": "^1.0.0"` inside a list `[...]` instead of `"pkg ^1.0.0"`.
    """
    containers: list[str] = []
    output: list[str] = []
    pos = 0
    length = len(raw_json)

    while pos < length:
        token = raw_json[pos]

        if token in ("[", "{"):
            containers.append(token)
            output.append(token)
            pos += 1
        elif token == "]":
            if containers and containers[-1] == "[":
                containers.pop()
            output.append(token)
            pos += 1
        elif token == "}":
            if containers and containers[-1] == "{":
                containers.pop()
            output.append(token)
            pos += 1
        elif containers and containers[-1] == "[" and token == '"':
            closing_quote = raw_json.find('"', pos + 1)
            if closing_quote == -1:
                output.append(token)
                pos += 1
                continue

            quoted_key = raw_json[pos : closing_quote + 1]
            lookahead = raw_json[closing_quote + 1 : closing_quote + 10].lstrip()

            if lookahead.startswith(":"):
                colon_pos = raw_json.index(":", closing_quote + 1)
                rhs = raw_json[colon_pos + 1 :].lstrip()
                if rhs.startswith('"'):
                    val_open = raw_json.index('"', colon_pos + 1)
                    val_close = raw_json.find('"', val_open + 1)
                    if val_close != -1:
                        val_str = raw_json[val_open + 1 : val_close]
                        clean_key = quoted_key[1:-1]
                        output.append(f'"{clean_key} {val_str}"')
                        pos = val_close + 1
                        continue

            output.append(quoted_key)
            pos = closing_quote + 1
        else:
            output.append(token)
            pos += 1

    return "".join(output)


def _strip_fences(raw_text: str) -> str:
    """Remove surrounding markdown code fences if present."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse a JSON dictionary from an LLM response string, automatically recovering
    from markdown fences, trailing commas, and key-value pairs inside lists.
    """
    cleaned = _strip_fences(raw_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        snippet = brace_match.group(0)
        snippet_no_trailing = re.sub(r",\s*([}\]])", r"\1", snippet)

        try:
            return json.loads(snippet_no_trailing)
        except json.JSONDecodeError:
            pass

        repaired = _normalize_array_key_value_pairs(snippet_no_trailing)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Unable to parse JSON object from model output", raw_text, 0)
