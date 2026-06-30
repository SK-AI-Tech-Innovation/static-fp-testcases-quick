// ACE-EXPECT: detect
// CATEGORY: should_detect/prompt_few_shot
// LANGUAGE: java
// ISSUE: A complex multi-field entity-extraction prompt (name, date, amount, category) is sent with only an instruction and zero few-shot examples, hurting format consistency and accuracy.
// EXPECTED-FINDING: Complex extraction prompt with no in-context examples (zero-shot) where few-shot demonstrations are warranted.
// EXPECTED-FIX: Add 2-3 representative input/output example pairs to the prompt to anchor format and edge-case handling.
// SEVERITY-HINT: suggestion
/** Zero-shot extraction of structured fields from messy receipt text. */
package fp.testcases;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Model;

public class JavaPromptFewShot1 {

    private final AnthropicClient client = AnthropicOkHttpClient.fromEnv();

    public String extractFields(String receiptText) {
        // Complex multi-field extraction, but the prompt gives zero examples to follow.
        String prompt = "Extract the merchant name, transaction date (YYYY-MM-DD), "
                + "total amount as a decimal, and spending category from the receipt below. "
                + "Return one field per line as key: value.\n\nReceipt:\n" + receiptText;

        MessageCreateParams params = MessageCreateParams.builder()
                .model(Model.CLAUDE_OPUS_4_20250514)
                .maxTokens(512)
                .addUserMessage(prompt)
                .build();

        Message message = client.messages().create(params);
        return message.content().get(0).text().orElseThrow().text();
    }

    public static void main(String[] args) {
        JavaPromptFewShot1 app = new JavaPromptFewShot1();
        System.out.println(app.extractFields("STARMART 03/14/25 groceries $52.10 paid visa"));
    }
}
