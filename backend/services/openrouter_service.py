# openrouter_service.py

"""
This module interfaces with the OpenRouter API to perform contextual question answering.
It is designed for use in an LLM-Powered Intelligent Query–Retrieval System that processes large documents and
provides concise, context-aware answers. Tailored for real-world use cases in insurance, legal, HR, and compliance domains.
"""

import requests
import os
from core.config import key

API_KEY = key.oss_api_key
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# You can change the model here depending on your need
DEFAULT_MODEL = "openai/gpt-oss-120b"

def answer_question_with_context(context: str, question: str, model: str = DEFAULT_MODEL) -> str:
    """
    Sends a prompt to the OpenRouter model with contextual information and a question.
    The LLM responds concisely, drawing only from the provided context.

    Args:
        context (str): Textual data extracted from documents.
        question (str): The user query.
        model (str): LLM model ID to use (default: Mixtral).

    Returns:
        str: The model's concise and contextual answer.
    """

    # Prompt designed to guide the LLM for concise and domain-relevant output
    system_prompt = (
    "You are an intelligent and context-aware assistant. "
    "Make the answers crisp without removing any data."
    "Make the answers sound professional and smart."
    "The answers should be simple paragraphs. Do not add any unnecesarry special characters like '*' or escape or newline tags without need."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",  # Replace with your site
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post(BASE_URL, headers=headers, json=body)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}\n\nRaw response: {response.text}"
