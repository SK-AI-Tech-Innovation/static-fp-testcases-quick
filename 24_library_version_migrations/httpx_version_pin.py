# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: openai-python + httpx — handling the httpx `proxies`-removal version quirk
# WHY-CORRECT: httpx removed the `proxies` argument from Client in 0.28, which broke older
#   openai SDK versions that passed it. Constructing an explicit httpx.Client and handing it
#   to the OpenAI client via http_client= is the documented workaround. The version-aware
#   branch (mounts vs. proxies) is correct defensive code, not a smell.
# EXPECTED-WRONG: engine may flag the manual httpx.Client construction as unnecessary, flag
#   the version comparison as a code smell, or claim http_client= is not a valid OpenAI kwarg.
# CORRECT-VERDICT: no findings
"""Build an OpenAI client with an explicit httpx.Client to sidestep an httpx version quirk."""
import os

import httpx
from openai import OpenAI

PROXY_URL = os.environ.get("HTTPS_PROXY")


def build_http_client() -> httpx.Client:
    # httpx >= 0.28 dropped the `proxies` kwarg in favor of `mounts`.
    httpx_version = tuple(int(p) for p in httpx.__version__.split(".")[:2])
    if PROXY_URL is None:
        return httpx.Client(timeout=30.0)
    if httpx_version >= (0, 28):
        transport = httpx.HTTPTransport(proxy=PROXY_URL)
        return httpx.Client(mounts={"all://": transport}, timeout=30.0)
    return httpx.Client(proxies=PROXY_URL, timeout=30.0)


def make_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=build_http_client(),
    )


if __name__ == "__main__":
    client = make_openai_client()
    print(type(client).__name__)
