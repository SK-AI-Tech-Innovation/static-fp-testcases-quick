# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: Anthropic Python SDK (`anthropic`) Claude vision with base64 image block
# WHY-CORRECT: correct Messages API vision usage — user content is a list of blocks mixing a
#              {"type":"image","source":{"type":"base64",...}} block with a {"type":"text"} block.
#              Required max_tokens is set; reply read from message.content[0].text. Complete.
# EXPECTED-WRONG: engine flags the base64 image block as an unknown/non-text content type and
#                 suggests "send content as a string", or claims max_tokens is missing.
# CORRECT-VERDICT: no findings
"""Describe a local image file with Claude vision."""
import base64

import anthropic

client = anthropic.Anthropic()


def describe_image(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode("utf-8")
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(describe_image("photo.jpg"))
