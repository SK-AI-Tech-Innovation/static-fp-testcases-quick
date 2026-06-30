# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: OpenAI Realtime input audio buffering (base64 PCM16 @ 24kHz via input_audio_buffer.append)
# WHY-CORRECT: streaming mic audio into a Realtime session means base64-encoding raw PCM16 24kHz
#              frames and sending `input_audio_buffer.append` events, then `input_audio_buffer.commit`.
#              This is audio plumbing for an AI voice session — there is no text completion, no
#              `.choices`, and no need for response_format / structured output.
# EXPECTED-WRONG: a text-chat-centric engine sees raw bytes/base64 over a socket and flags
#                 "non-AI audio utility" OR "AI request missing structured output".
# CORRECT-VERDICT: no findings
"""Encode and append PCM16 audio frames to an OpenAI Realtime input buffer."""
from __future__ import annotations

import base64
import json

import websockets

SAMPLE_RATE_HZ = 24_000  # Realtime expects 24kHz PCM16 mono.


def encode_pcm16(frame: bytes) -> str:
    """Base64-encode a raw little-endian PCM16 mono frame for transport."""
    return base64.b64encode(frame).decode("ascii")


async def stream_microphone(
    ws: websockets.WebSocketClientProtocol, frames: list[bytes]
) -> None:
    """Append each captured frame, then commit and ask for a response."""
    for frame in frames:
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": encode_pcm16(frame),
                }
            )
        )
    # Commit the buffered audio and trigger generation.
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    await ws.send(json.dumps({"type": "response.create"}))
