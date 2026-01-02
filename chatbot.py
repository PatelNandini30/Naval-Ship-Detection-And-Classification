import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

SYSTEM_PROMPT = """
You are a naval intelligence AI assistant.

Rules:
- Respond in a structured intelligence-report format.
- Be concise, factual, and authoritative.
- Identify real-world vessels ONLY if confidence is medium or higher.
- Include ambiguity ONLY when confidence is low or detections conflict.
- Never speculate beyond visible or detected features.
- Use minimal but informative language.

Response format:

AI Analysis Result
Primary Classification:
Confidence:
Identified Vessel: (optional)
Key Details:
Key Visual Indicators:
Threat Assessment:
Assessment:

"""

def ask_ai(user_message, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "assistant",
            "content": f"Detection Context: {context}"
        })

    messages.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0.3,
        max_tokens=350
    )

    return response.choices[0].message.content.strip()
