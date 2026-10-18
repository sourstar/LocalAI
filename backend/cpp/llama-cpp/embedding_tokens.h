#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace llama_grpc {

// Token ids sent to Embedding (PredictOptions.EmbeddingTokens, which
// core/backend/embeddings.go fills when /v1/embeddings is given tokens) are an
// input's tokens as TokenizeString returns them: without special tokens. A text
// is embedded with the special tokens its vocabulary adds (BOS/EOS, CLS/SEP, or
// none at all), so the ids get the same ones: where a probe text's own tokens
// (`bare`) sit inside its tokens with the special ones added (`full`), the ids
// take their place. Embedding a text's tokens is then embedding the text.
//
// nullopt when `bare` is empty or does not occur inside `full`: there is then
// no telling where the special tokens go.
inline std::optional<std::vector<int32_t>> with_special_tokens(const std::vector<int32_t> & bare,
                                                               const std::vector<int32_t> & full,
                                                               const std::vector<int32_t> & ids) {
    if (bare.empty()) {
        return std::nullopt;
    }
    const auto at = std::search(full.begin(), full.end(), bare.begin(), bare.end());
    if (at == full.end()) {
        return std::nullopt;
    }
    std::vector<int32_t> out(full.begin(), at);
    out.insert(out.end(), ids.begin(), ids.end());
    out.insert(out.end(), at + static_cast<std::ptrdiff_t>(bare.size()), full.end());
    return out;
}

} // namespace llama_grpc
