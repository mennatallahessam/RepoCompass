"""
Common LLM initialization, API key management, and resilient JSON extraction utilities for RepoCompass.
"""
import json
import os
import re
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def has_active_api_key() -> bool:
    """Return True if a user-supplied OpenAI API key is configured and non-empty."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return False
    # Ignore the expired legacy placeholder JWT from the original template
    if key.startswith("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJhZXNKN2kxNGNidnVuTU40MTJr"):
        return False
    return True


def set_runtime_api_config(api_key: str, base_url: Optional[str] = None) -> None:
    """Configure runtime OpenAI API credentials from UI or environment."""
    cleaned_key = api_key.strip()
    if cleaned_key:
        os.environ["OPENAI_API_KEY"] = cleaned_key
    if base_url and base_url.strip():
        os.environ["OPENAI_BASE_URL"] = base_url.strip()
    elif "quantumblack.com" in os.environ.get("OPENAI_BASE_URL", ""):
        os.environ.pop("OPENAI_BASE_URL", None)


def create_chat_model(temperature: float = 0.1, model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "sk-placeholder-offline-mode"
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    if base_url and "quantumblack.com" in base_url and not has_active_api_key():
        base_url = None
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)


def invoke_llm_text(llm: ChatOpenAI, prompt: str) -> str:
    """Invoke the language model and return its string content, raising if offline/unconfigured."""
    if not has_active_api_key():
        raise RuntimeError("No active OpenAI API key configured; using built-in static analysis engine.")
    # Recreate client with fresh env vars in case user updated key in sidebar
    active_llm = create_chat_model(
        temperature=getattr(llm, "temperature", 0.1) or 0.1,
        model_name=getattr(llm, "model_name", "gpt-4o-mini") or "gpt-4o-mini",
    )
    reply = active_llm.invoke(prompt)
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
