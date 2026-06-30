// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: Anthropic TypeScript SDK Messages API with tool use (`client.messages.create({ tools })`)
// WHY-CORRECT: canonical Anthropic agentic loop in TS — declare tools, detect `stop_reason === "tool_use"`,
//              run the tool, and return a `tool_result` content block. The loop and typing are complete.
// EXPECTED-WRONG: a Python+OpenAI-centric engine doesn't recognize the TS Anthropic SDK; it may flag
//                 "no structured output / response_format missing" or mis-read tool_use blocks as raw text.
// CORRECT-VERDICT: no findings
/** Anthropic TS Messages API agentic loop with a single stock-price tool. */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const tools: Anthropic.Tool[] = [
  {
    name: "get_stock_price",
    description: "Get the latest price for a ticker symbol in USD.",
    input_schema: {
      type: "object",
      properties: { ticker: { type: "string" } },
      required: ["ticker"],
    },
  },
];

function getStockPrice(ticker: string): number {
  return 187.42; // stand-in for a real market-data lookup
}

export async function askWithTools(question: string): Promise<string> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: question }];

  while (true) {
    const resp = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 1024,
      tools,
      messages,
    });

    if (resp.stop_reason !== "tool_use") {
      return resp.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    messages.push({ role: "assistant", content: resp.content });
    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    for (const block of resp.content) {
      if (block.type === "tool_use") {
        const { ticker } = block.input as { ticker: string };
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: JSON.stringify({ price: getStockPrice(ticker) }),
        });
      }
    }
    messages.push({ role: "user", content: toolResults });
  }
}
