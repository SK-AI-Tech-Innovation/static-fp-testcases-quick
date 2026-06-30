# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: Azure OpenAI in v0.x was configured via module globals: openai.api_type="azure",
#   openai.api_base, openai.api_version, and the deployment selected with deployment_id= (or
#   engine=). This is exactly the documented Azure pattern for v0.x; the api_version string
#   shown was a real GA version.
# EXPECTED-WRONG: engine may flag the global config / deployment_id as deprecated and push the
#   v1 AzureOpenAI() client (azure_endpoint=, azure_deployment=), or call the globals a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x Azure OpenAI configuration via module globals and deployment_id."""
import os

import openai

openai.api_type = "azure"
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
openai.api_version = "2023-05-15"
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]


def chat(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        deployment_id="gpt-35-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat("Hello from Azure."))
