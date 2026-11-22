// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace llama_grpc {

// llama.cpp's server can let a request give the context back after its answer,
// keeping the model loaded (--context-on-demand): a request that carries
// keep_context_ms has the context freed that many milliseconds after it, and
// the next request creates it again - requests back to back carry a keep and
// do not pay a creation each. A request that carries none uses the context as
// it always did: it is not freed after it. LocalAI carries the request's keep
// in its metadata map; the model option of the same name gives one to the
// requests that carry none.

// parse_keep_context_ms converts the value of the keep_context_ms metadata key,
// or of the model option of the same name, to milliseconds. It returns -1
// ("not said": the context stays loaded after the request, unless the model
// option says otherwise) for anything that is not a whole, non-negative number
// of milliseconds that fits in int32_t.
inline int32_t parse_keep_context_ms(const std::string & value) {
    try {
        size_t used = 0;
        const long long ms = std::stoll(value, &used);
        if (used != value.size() || ms < 0 || ms > std::numeric_limits<int32_t>::max()) {
            return -1;
        }
        return static_cast<int32_t>(ms);
    } catch (const std::exception &) {
        return -1;
    }
}

// keep_context_ms_from reads the key from a metadata map (any map from string
// to string), -1 when the map does not carry it.
template <typename Map>
int32_t keep_context_ms_from(const Map & metadata) {
    const auto it = metadata.find("keep_context_ms");
    return it == metadata.end() ? -1 : parse_keep_context_ms(it->second);
}

// The fields below exist only in llama.cpp trees that have the feature. They
// are detected, so that this source still builds against pins and forks
// without it; the setters return whether the field was there to set.
template <typename T, typename = void>
struct has_keep_context_ms : std::false_type {};
template <typename T>
struct has_keep_context_ms<T, std::void_t<decltype(std::declval<T &>().keep_context_ms)>> : std::true_type {};

template <typename T, typename = void>
struct has_context_on_demand : std::false_type {};
template <typename T>
struct has_context_on_demand<T, std::void_t<decltype(std::declval<T &>().context_on_demand)>> : std::true_type {};

// set_keep_context_ms sets the keep of a task's params, or the default keep of
// the server's params.
template <typename Params>
bool set_keep_context_ms(Params & params, int32_t ms) {
    if constexpr (has_keep_context_ms<Params>::value) {
        params.keep_context_ms = ms;
        return true;
    } else {
        (void) params;
        (void) ms;
        return false;
    }
}

// set_context_on_demand switches the feature on the server's params.
template <typename Params>
bool set_context_on_demand(Params & params, bool on) {
    if constexpr (has_context_on_demand<Params>::value) {
        params.context_on_demand = on;
        return true;
    } else {
        (void) params;
        (void) on;
        return false;
    }
}

template <typename T, typename = void>
struct has_context_canary : std::false_type {};
template <typename T>
struct has_context_canary<T, std::void_t<decltype(std::declval<T &>().context_canary)>> : std::true_type {};

// set_context_canary switches the context canary on the server's params: with
// it (llama.cpp's default) each context created on demand must give the output
// the load's context gave for a fixed input, or it is discarded and the
// request refused.
template <typename Params>
bool set_context_canary(Params & params, bool on) {
    if constexpr (has_context_canary<Params>::value) {
        params.context_canary = on;
        return true;
    } else {
        (void) params;
        (void) on;
        return false;
    }
}

// How llama.cpp's server words the refusal of a request whose newly created
// context did not reproduce the canary, and the trailing metadata under which
// the backend says so beside the error: a caller can tell such a refusal
// (retry; a context was discarded) from memory that could not be had.
inline constexpr const char * CONTEXT_CANARY_FAILED  = "context canary failed";
inline constexpr const char * CONTEXT_CANARY_TRAILER = "llama-context-canary";

inline bool is_context_canary_failure(const std::string & message) {
    return message.rfind(CONTEXT_CANARY_FAILED, 0) == 0;
}

// context_on_demand_of reads the switch from the server's params; false for a
// llama.cpp without the feature.
template <typename Params>
bool context_on_demand_of(const Params & params) {
    if constexpr (has_context_on_demand<Params>::value) {
        return params.context_on_demand;
    } else {
        (void) params;
        return false;
    }
}

template <typename T, typename = void>
struct has_t_ctx_create_us : std::false_type {};
template <typename T>
struct has_t_ctx_create_us<T, std::void_t<decltype(std::declval<T &>().t_ctx_create_us)>> : std::true_type {};

// ctx_create_us_of reads from a task's result the microseconds it took to
// create the context for that task; negative when the context was already
// there, or when this llama.cpp does not say.
template <typename Result>
int64_t ctx_create_us_of(const Result & result) {
    if constexpr (has_t_ctx_create_us<Result>::value) {
        return result.t_ctx_create_us;
    } else {
        (void) result;
        return -1;
    }
}

// The trailing metadata key under which an answer of a model loaded with
// context_on_demand says whether the context was created for that call: the
// microseconds its creation took, or -1 when the context was already there.
// It is the backend's own word, per call, for whoever counts creations (a
// caller cannot tell one from its own clock: the context may have been kept
// for another caller's request).
inline constexpr const char * CONTEXT_CREATED_TRAILER = "llama-context-created-us";

// context_created_us folds the results of one call: the sum of the creations
// its tasks paid for, or -1 when none did.
template <typename Results>
int64_t context_created_us(const Results & results) {
    int64_t total = -1;
    for (const auto & result : results) {
        const int64_t us = ctx_create_us_of(*result);
        if (us >= 0) {
            total = (total < 0 ? 0 : total) + us;
        }
    }
    return total;
}

} // namespace llama_grpc
