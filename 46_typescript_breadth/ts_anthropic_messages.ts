// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: Anthropic TypeScript SDK Messages API (`client.messages.create`)
// WHY-CORRECT: canonical Anthropic TS call — `model`, required `max_tokens`, a top-level `system` prompt,
//              and a `messages` array. `claude-opus-4-8` is a current id. `max_tokens` is mandatory for the
//              Anthropic API (it is NOT the deprecated OpenAI `max_tokens`).
// EXPECTED-WRONG: an OpenAI-centric engine may flag `max_tokens` as "deprecated, use max_completion_tokens"
//                 or treat the top-level `system` field as missing a system message — both wrong for Anthropic.
// CORRECT-VERDICT: no findings
/** Single-turn Anthropic completion with a system prompt. */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

export async function rewrite(text: string, tone: string): Promise<string> {
  const message = await client.messages.create({
    model: "claude-opus-4-8",
    max_tokens: 1024,
    system: `Rewrite the user's text in a ${tone} tone. Preserve meaning.`,
    messages: [{ role: "user", content: text }],
  });

  return message.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}
