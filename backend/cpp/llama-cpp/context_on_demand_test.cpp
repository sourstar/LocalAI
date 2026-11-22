// SPDX-License-Identifier: MIT

#include "context_on_demand.h"

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

struct params_with    { int32_t keep_context_ms = -1; bool context_on_demand = false; bool context_canary = true; };
struct params_without { int unrelated = 0; };
struct result_with    { int64_t t_ctx_create_us = -1; };
struct result_without { int unrelated = 0; };

} // namespace

int main() {
    using namespace llama_grpc;

    // a whole, non-negative number of milliseconds
    CHECK(parse_keep_context_ms("0") == 0);
    CHECK(parse_keep_context_ms("20") == 20);
    CHECK(parse_keep_context_ms("300000") == 300000);
    CHECK(parse_keep_context_ms("2147483647") == 2147483647);

    // anything else is "not said"
    CHECK(parse_keep_context_ms("") == -1);
    CHECK(parse_keep_context_ms("-1") == -1);
    CHECK(parse_keep_context_ms("20ms") == -1);
    CHECK(parse_keep_context_ms("1.5") == -1);
    CHECK(parse_keep_context_ms("true") == -1);
    CHECK(parse_keep_context_ms("2147483648") == -1);
    CHECK(parse_keep_context_ms("99999999999999999999999") == -1);

    // the request's metadata
    const std::map<std::string, std::string> none  = {{"chat_template_kwargs", "{}"}};
    const std::map<std::string, std::string> some  = {{"keep_context_ms", "20"}};
    const std::map<std::string, std::string> wrong = {{"keep_context_ms", "soon"}};
    CHECK(keep_context_ms_from(none) == -1);
    CHECK(keep_context_ms_from(some) == 20);
    CHECK(keep_context_ms_from(wrong) == -1);

    // a llama.cpp with the fields, and one without
    params_with with;
    CHECK(set_keep_context_ms(with, 20));
    CHECK(with.keep_context_ms == 20);
    CHECK(set_context_on_demand(with, true));
    CHECK(with.context_on_demand);

    params_without without;
    CHECK(!set_keep_context_ms(without, 20));
    CHECK(!set_context_on_demand(without, true));
    CHECK(context_on_demand_of(with));
    CHECK(!context_on_demand_of(without));

    // the canary: on unless switched off, and a llama.cpp without it says so
    CHECK(with.context_canary);
    CHECK(set_context_canary(with, false));
    CHECK(!with.context_canary);
    CHECK(!set_context_canary(without, false));
    CHECK(is_context_canary_failure("context canary failed: the context created for this request did not reproduce"));
    CHECK(!is_context_canary_failure("the context cannot be created now, the memory it needs may be in use"));
    CHECK(!is_context_canary_failure("a context canary failed"));

    // a result says when the context was created for its task
    result_with created;
    created.t_ctx_create_us = 1500;
    CHECK(ctx_create_us_of(created) == 1500);
    CHECK(ctx_create_us_of(result_with()) == -1);
    CHECK(ctx_create_us_of(result_without()) == -1);

    // a call's results: the creations its tasks paid for, -1 when none did
    std::vector<std::unique_ptr<result_with>> results;
    results.push_back(std::make_unique<result_with>());
    results.push_back(std::make_unique<result_with>());
    CHECK(context_created_us(results) == -1);
    results[1]->t_ctx_create_us = 0;
    CHECK(context_created_us(results) == 0);
    results[0]->t_ctx_create_us = 1500;
    CHECK(context_created_us(results) == 1500);
    std::vector<std::unique_ptr<result_without>> silent;
    silent.push_back(std::make_unique<result_without>());
    CHECK(context_created_us(silent) == -1);

    return 0;
}
