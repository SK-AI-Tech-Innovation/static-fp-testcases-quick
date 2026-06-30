// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: c
// SOURCE: llama.cpp (llama.h) local inference
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal greedy generation loop against a local GGUF model using llama.cpp.
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = argv[2];

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    // Tokenize the prompt.
    const int n_prompt = -llama_tokenize(vocab, prompt, (int) strlen(prompt),
                                         NULL, 0, true, true);
    llama_token *tokens = malloc(sizeof(llama_token) * n_prompt);
    if (llama_tokenize(vocab, prompt, (int) strlen(prompt),
                       tokens, n_prompt, true, true) < 0) {
        fprintf(stderr, "tokenization failed\n");
        free(tokens);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    struct llama_batch batch = llama_batch_get_one(tokens, n_prompt);

    struct llama_sampler *smpl = llama_sampler_init_greedy();

    const int n_predict = 64;
    for (int i = 0; i < n_predict; i++) {
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            break;
        }

        llama_token next = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
        if (n > 0) {
            fwrite(buf, 1, (size_t) n, stdout);
            fflush(stdout);
        }

        batch = llama_batch_get_one(&next, 1);
    }

    printf("\n");

    llama_sampler_free(smpl);
    free(tokens);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
