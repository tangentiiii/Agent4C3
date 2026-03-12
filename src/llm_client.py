import json
import os
import time
import yaml
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _client


def load_prompt(prompt_name: str) -> dict:
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.yml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5-mini",
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    json_mode: bool = False,
) -> str:
    client = get_client()

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content (possibly filtered)")
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"LLM call failed (attempt {attempt + 1}): {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    **kwargs,
) -> dict | list:
    raw = call_llm(system_prompt, user_prompt, json_mode=True, **kwargs)
    return json.loads(raw)
