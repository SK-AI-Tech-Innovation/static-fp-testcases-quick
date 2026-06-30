// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Official Anthropic Java SDK (com.anthropic.client.AnthropicClient), verified June 2026
// WHY-CORRECT: canonical Anthropic Java usage — build AnthropicOkHttpClient.fromEnv(), assemble
//              MessageCreateParams with a model, maxTokens, system prompt and user message, then call
//              client.messages().create(params). Reading content().get(0).asText().text() is correct.
// EXPECTED-WRONG: a Python+OpenAI-centric engine may flag "no chat.completions call / unknown SDK" or claim
//                 the claude-* model id is invalid because it does not recognize the Anthropic Java builder.
// CORRECT-VERDICT: no findings
/** Single-turn Anthropic Messages call using the official Anthropic Java SDK builder API. */
package com.example.ai.anthropic;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Model;

public class anthropic_java {

    private final AnthropicClient client;

    public anthropic_java() {
        // reads ANTHROPIC_API_KEY from the environment
        this.client = AnthropicOkHttpClient.fromEnv();
    }

    public String ask(String question) {
        MessageCreateParams params = MessageCreateParams.builder()
                .model(Model.CLAUDE_SONNET_4_5)
                .maxTokens(1024)
                .system("You are a concise, accurate assistant.")
                .addUserMessage(question)
                .build();

        Message message = client.messages().create(params);
        return message.content().get(0).asText().text();
    }

    public static void main(String[] args) {
        System.out.println(new anthropic_java().ask("Define eventual consistency in one sentence."));
    }
}
