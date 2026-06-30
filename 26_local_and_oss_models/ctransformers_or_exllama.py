# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: ctransformers local GGUF inference (callable model)
# WHY-CORRECT: ctransformers' AutoModelForCausalLM loads a quantized GGUF and is called directly as
#              a function to stream/generate tokens locally. GPU offload via gpu_layers, sampling via
#              temperature/top_p. No server, endpoint, or key — pure on-device generation.
# EXPECTED-WRONG: engine doesn't recognize the callable-model generation as LLM usage (non-AI), or
#                 maps the AutoModelForCausalLM name onto a hosted pattern and asks for API retries /
#                 timeouts that have no meaning for a local quantized model call.
# CORRECT-VERDICT: no findings
"""Local quantized inference with ctransformers."""
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=50,
    context_length=2048,
)


def generate(prompt: str) -> str:
    return llm(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
    )


if __name__ == "__main__":
    print(generate("[INST] Recommend a book about space exploration. [/INST]"))
