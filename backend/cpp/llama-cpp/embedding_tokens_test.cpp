#include "embedding_tokens.h"

#include <cstdio>
#include <vector>

using llama_grpc::with_special_tokens;
using ids_t = std::vector<int32_t>;

static int expect(const char * what, const std::optional<ids_t> & got, const ids_t & want) {
    if (!got || *got != want) {
        std::fprintf(stderr, "%s: the special tokens were not placed as for the text\n", what);
        return 1;
    }
    return 0;
}

int main() {
    const ids_t ids = {7, 8, 9};
    int failed = 0;
    // BOS and EOS (an XLM-RoBERTa vocabulary: <s> ... </s>)
    failed += expect("bos+eos", with_special_tokens({5}, {0, 5, 2}, ids), {0, 7, 8, 9, 2});
    // BOS only (a llama vocabulary)
    failed += expect("bos", with_special_tokens({5}, {1, 5}, ids), {1, 7, 8, 9});
    // EOS only: it follows the ids, never precedes them
    failed += expect("eos", with_special_tokens({5}, {5, 2}, ids), {7, 8, 9, 2});
    // none: the ids as they are
    failed += expect("none", with_special_tokens({5}, {5}, ids), ids);
    // a probe of several tokens
    failed += expect("long probe", with_special_tokens({5, 6}, {101, 5, 6, 102}, ids),
                     {101, 7, 8, 9, 102});
    if (with_special_tokens({5}, {0, 6, 2}, ids) || with_special_tokens({}, {0, 2}, ids)) {
        std::fprintf(stderr, "special tokens were placed without the probe's own tokens to go by\n");
        failed += 1;
    }
    return failed == 0 ? 0 : 1;
}
