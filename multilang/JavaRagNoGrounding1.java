// ACE-EXPECT: detect
// CATEGORY: should_detect/rag_no_grounding
// LANGUAGE: java
// ISSUE: Documents are retrieved from an embedding store but never injected into the prompt; the model answers from parametric memory only, with no grounding, citations, or abstain path.
// EXPECTED-FINDING: RAG flow retrieves context but discards it - the LLM prompt contains only the question, so the answer is ungrounded with no citations or abstain behavior.
// EXPECTED-FIX: Inject the retrieved passages into the prompt, instruct the model to answer only from context, cite sources, and abstain when context is insufficient.
// SEVERITY-HINT: warning
/** Retrieves passages then ignores them when prompting the model. */
package fp.testcases;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;

import java.util.List;

public class JavaRagNoGrounding1 {

    private final OpenAIClient client = OpenAIOkHttpClient.fromEnv();
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;

    public JavaRagNoGrounding1(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> store) {
        this.embeddingModel = embeddingModel;
        this.store = store;
    }

    public String ask(String question) {
        Embedding queryEmbedding = embeddingModel.embed(question).content();
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(5)
                .build();
        List<EmbeddingMatch<TextSegment>> matches = store.search(request).matches();
        // matches retrieved... and then completely ignored below.

        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O)
                .addUserMessage(question)
                .build();
        ChatCompletion completion = client.chat().completions().create(params);
        return completion.choices().get(0).message().content().orElse("");
    }
}
