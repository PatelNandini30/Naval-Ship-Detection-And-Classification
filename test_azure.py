from dotenv import load_dotenv
from openai import AzureOpenAI
import os

load_dotenv()

print("ENDPOINT =", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("DEPLOYMENT =", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print("VERSION =", os.getenv("AZURE_OPENAI_API_VERSION"))

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

resp = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=[{"role": "user", "content": "Hello, confirm you are working"}]
)

print(resp.choices[0].message.content)
