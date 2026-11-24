// SPDX-License-Identifier: MIT

#include <cstdio>
#include <map>
#include <string>

#include "tts_request_options.h"

static int failures = 0;

static void check(bool ok, const char * name) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", name);
    }
}

static void test_accepts_a_minimal_valid_request() {
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "en", {}, true);

    check(opts.ok, "minimal request is accepted");
    check(opts.error.empty(), "minimal request has no error");
    check(opts.text == "Hello world", "text passes through");
    check(opts.voice_path == "/models/voices/ref.wav", "voice path passes through");
    check(opts.voice.empty(), "reference mode leaves the named voice empty");
    check(opts.language == "en", "language passes through");
    check(opts.top_k == 0, "top_k defaults to the unset sentinel");
    check(opts.top_p == 0.0f, "top_p defaults to the unset sentinel");
    check(opts.max_frames == 0, "max_frames defaults to the unset sentinel");
}

static void test_rejects_empty_text() {
    const auto opts = llama_grpc::parse_tts_request_options(
        "", "/models/voices/ref.wav", "en", {}, true);

    check(!opts.ok, "empty text is rejected");
    check(opts.error.find("text") != std::string::npos, "empty-text error names the field");

    const auto no_ref = llama_grpc::parse_tts_request_options(
        "", "Female", "en", {}, false);
    check(!no_ref.ok, "empty text is rejected without a speaker reference too");
}

static void test_rejects_missing_speaker_reference() {
    // Qwen3-TTS Base has no built-in speaker; without a reference it produces
    // an arbitrary voice, so this must be a hard error rather than a surprise.
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "", "en", {}, true);

    check(!opts.ok, "missing voice is rejected");
    check(opts.error.find("voice") != std::string::npos, "missing-voice error names the field");
}

static void test_accepts_named_voice_without_reference() {
    // LFM3-Audio has no speaker reference. Its "tts" chat template takes a
    // voice name ("female" or "male"), so the value is a name, not a path.
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "male", "en", {}, false);

    check(opts.ok, "named voice is accepted when no reference is needed");
    check(opts.error.empty(), "named voice has no error");
    check(opts.voice == "male", "named voice passes through");
    check(opts.voice_path.empty(), "named voice is not treated as a file path");
    check(opts.language == "en", "language passes through without a reference");
}

static void test_accepts_empty_voice_without_reference() {
    // An empty voice lets the template pick its own default.
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "", "", {}, false);

    check(opts.ok, "empty voice is accepted when no reference is needed");
    check(opts.voice.empty(), "empty voice stays empty for the template default");
    check(opts.voice_path.empty(), "empty voice without reference has no path");
}

static void test_a_model_that_clones_and_describes_takes_either_or_neither() {
    using mode = llama_grpc::tts_voice_mode;
    const std::map<std::string, std::string> params{{"ref_text", "Good morning."}};

    // A voice profile: the HTTP layer resolved it to a file and its transcript.
    const auto cloned = llama_grpc::parse_tts_request_options(
        "Hello world", "/voices/ref.wav", "en", "", params, mode::reference_or_name, true);
    check(cloned.ok, "a file is accepted as a reference");
    check(cloned.voice_path == "/voices/ref.wav", "the file is the reference");
    check(cloned.ref_text == "Good morning.", "the reference's transcript passes through");
    check(cloned.voice.empty(), "a reference leaves the described voice empty");

    // Not a file: a description for the "tts" template.
    const auto described = llama_grpc::parse_tts_request_options(
        "Hello world", "female, young adult", "en", "", params, mode::reference_or_name, false);
    check(described.ok, "a description is accepted");
    check(described.voice == "female, young adult", "the description is the template's voice");
    check(described.voice_path.empty(), "a description is not a file");
    check(described.ref_text.empty(), "a transcript without a reference is dropped");

    // Neither: the model's own voice (a warm-up call, a client that names none).
    const auto none = llama_grpc::parse_tts_request_options(
        "Hello world", "", "", "", {}, mode::reference_or_name, false);
    check(none.ok, "no voice is accepted when the reference is optional");
    check(none.voice.empty() && none.voice_path.empty(), "no voice stays none");

    // OpenAI's `instructions` describes the voice when the request names none,
    // beside a reference too.
    const auto instructed = llama_grpc::parse_tts_request_options(
        "Hello world", "", "en", "male, elderly", {}, mode::reference_or_name, false);
    check(instructed.voice == "male, elderly", "instructions describe the voice");
    const auto both = llama_grpc::parse_tts_request_options(
        "Hello world", "/voices/ref.wav", "en", "whisper", params, mode::reference_or_name, true);
    check(both.voice_path == "/voices/ref.wav" && both.voice == "whisper", "a reference and instructions go together");
    const auto named = llama_grpc::parse_tts_request_options(
        "Hello world", "male", "en", "whisper", {}, mode::name, false);
    check(named.voice == "male", "a named voice wins over instructions");
}

static void test_a_required_reference_keeps_its_rules_and_gains_the_transcript() {
    using mode = llama_grpc::tts_voice_mode;
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "/voices/ref.wav", "en", "ignored here", {{"ref_text", "Hi."}}, mode::reference, false);
    check(opts.ok && opts.voice_path == "/voices/ref.wav", "a required reference is taken as a path, checked when it is read");
    check(opts.ref_text == "Hi.", "a required reference carries its transcript");
    check(opts.voice.empty(), "instructions are not a voice for a model that only clones");

    const auto missing = llama_grpc::parse_tts_request_options(
        "Hello world", "", "en", "male", {}, mode::reference, false);
    check(!missing.ok, "a required reference is still required");
}

static void test_parses_sampling_params() {
    const std::map<std::string, std::string> params{
        {"top_k", "40"},
        {"top_p", "0.85"},
    };
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", params, true);

    check(opts.ok, "sampling params are accepted");
    check(opts.top_k == 40, "top_k is parsed");
    check(opts.top_p > 0.849f && opts.top_p < 0.851f, "top_p is parsed");
    check(opts.language.empty(), "absent language stays empty");
}

static void test_rejects_malformed_sampling_params() {
    const auto bad_top_k = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_k", "forty"}}, true);
    check(!bad_top_k.ok, "non-numeric top_k is rejected");
    check(bad_top_k.error.find("top_k") != std::string::npos, "top_k error names the field");

    const auto bad_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", ""}}, true);
    check(!bad_top_p.ok, "empty top_p is rejected");

    const auto trailing = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_k", "40abc"}}, true);
    check(!trailing.ok, "top_k with trailing garbage is rejected");

    const auto trailing_float = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "0.8abc"}}, true);
    check(!trailing_float.ok, "top_p with trailing garbage is rejected");

    // std::stol returns a long, which is wider than int32_t on 64-bit hosts, so
    // an in-range-for-long value still has to be caught before the narrowing.
    const auto overflow_top_k = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_k", "99999999999"}}, true);
    check(!overflow_top_k.ok, "top_k beyond int32 range is rejected");
    check(overflow_top_k.error.find("top_k") != std::string::npos,
          "top_k overflow error names the field");
}

static void test_rejects_out_of_range_sampling_params() {
    // These reach mtmd_helper::gen_audio::inp unconditionally downstream, where
    // the "> 0" sampler guard does not screen them, so they must die here.
    const auto negative_top_k = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_k", "-5"}}, true);
    check(!negative_top_k.ok, "negative top_k is rejected");
    check(negative_top_k.error.find("top_k") != std::string::npos,
          "negative top_k error names the field");

    const auto negative_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "-0.1"}}, true);
    check(!negative_top_p.ok, "negative top_p is rejected");
    check(negative_top_p.error.find("top_p") != std::string::npos,
          "negative top_p error names the field");

    const auto large_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "1.5"}}, true);
    check(!large_top_p.ok, "top_p above 1.0 is rejected");

    // NaN survives a naive "p < 0.0f || p > 1.0f" range test because every
    // comparison against NaN is false. This case pins the correct form.
    const auto nan_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "nan"}}, true);
    check(!nan_top_p.ok, "NaN top_p is rejected");

    const auto inf_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "inf"}}, true);
    check(!inf_top_p.ok, "infinite top_p is rejected");
}

static void test_sampling_validation_applies_without_reference() {
    // The range checks do not depend on the speaker-reference mode.
    const auto negative_top_k = llama_grpc::parse_tts_request_options(
        "Hello world", "female", "", {{"top_k", "-5"}}, false);
    check(!negative_top_k.ok, "negative top_k is rejected without a reference");

    const auto nan_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "female", "", {{"top_p", "nan"}}, false);
    check(!nan_top_p.ok, "NaN top_p is rejected without a reference");

    const auto max_frames = llama_grpc::parse_tts_request_options(
        "Hello world", "female", "", {{"max_frames", "120"}}, false);
    check(max_frames.ok && max_frames.max_frames == 120,
          "max_frames is parsed without a reference");
}

static void test_accepts_sampling_param_boundaries() {
    const auto zero_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "0.0"}}, true);
    check(zero_top_p.ok, "top_p of 0.0 is accepted");
    check(zero_top_p.top_p == 0.0f, "top_p of 0.0 round-trips");

    const auto one_top_p = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_p", "1.0"}}, true);
    check(one_top_p.ok, "top_p of 1.0 is accepted");
    check(one_top_p.top_p == 1.0f, "top_p of 1.0 round-trips");

    const auto zero_top_k = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_k", "0"}}, true);
    check(zero_top_k.ok, "top_k of 0 is accepted");
}

static void test_parses_max_frames() {
    // The consumer maps a positive value onto n_predict and leaves upstream's
    // 512-frame default in place when it is unset, so the sentinel matters as
    // much as the parsed value.
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", "120"}}, true);

    check(opts.ok, "max_frames is accepted");
    check(opts.max_frames == 120, "max_frames is parsed");

    const auto absent = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"top_k", "40"}}, true);
    check(absent.ok, "a request without max_frames is accepted");
    check(absent.max_frames == 0, "absent max_frames leaves the unset sentinel");

    const auto zero = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", "0"}}, true);
    check(zero.ok, "max_frames of 0 is accepted");
    check(zero.max_frames == 0, "max_frames of 0 means unset");
}

static void test_parses_the_decoding_s_seed_steps_and_guidance() {
    using mode = llama_grpc::tts_voice_mode;
    const auto unset = llama_grpc::parse_tts_request_options(
        "Hello world", "", "it", "", {}, mode::reference_or_name, false);
    check(unset.ok && unset.seed == -1 && unset.flow_steps == 0 && unset.cfg_scale == 0.0f,
          "seed, flow_steps and cfg_scale are unset when the request names none");

    const auto set = llama_grpc::parse_tts_request_options(
        "Hello world", "", "it", "", {{"seed", "0"}, {"flow_steps", "16"}, {"cfg_scale", "2.5"}},
        mode::reference_or_name, false);
    check(set.ok, "seed, flow_steps and cfg_scale are accepted");
    check(set.seed == 0, "a seed of 0 is a seed, not the unset sentinel");
    check(set.flow_steps == 16, "flow_steps is parsed");
    check(set.cfg_scale == 2.5f, "cfg_scale is parsed");

    const auto random = llama_grpc::parse_tts_request_options(
        "Hello world", "", "it", "", {{"seed", "-1"}}, mode::reference_or_name, false);
    check(random.ok && random.seed == -1, "a seed of -1 is upstream's random seed: unset");
}

static void test_rejects_malformed_decoding_params() {
    using mode = llama_grpc::tts_voice_mode;
    const std::pair<const char *, const char *> bad[] = {
        {"seed", "-2"}, {"seed", "zero"}, {"seed", "1.5"},
        {"flow_steps", "-1"}, {"flow_steps", "1025"}, {"flow_steps", "16x"},
        {"cfg_scale", "-0.5"}, {"cfg_scale", "nan"}, {"cfg_scale", "inf"}, {"cfg_scale", "high"},
    };
    for (const auto & [key, value] : bad) {
        const auto opts = llama_grpc::parse_tts_request_options(
            "Hello world", "", "it", "", {{key, value}}, mode::reference_or_name, false);
        check(!opts.ok, (std::string(key) + " of \"" + value + "\" is rejected").c_str());
        check(opts.error.find(key) != std::string::npos, (std::string(key) + "'s error names the field").c_str());
    }
}

static void test_rejects_malformed_max_frames() {
    const auto negative = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", "-1"}}, true);
    check(!negative.ok, "negative max_frames is rejected");
    check(negative.error.find("max_frames") != std::string::npos,
          "negative max_frames error names the field");

    const auto non_numeric = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", "many"}}, true);
    check(!non_numeric.ok, "non-numeric max_frames is rejected");
    check(non_numeric.error.find("max_frames") != std::string::npos,
          "non-numeric max_frames error names the field");

    const auto trailing = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", "120abc"}}, true);
    check(!trailing.ok, "max_frames with trailing garbage is rejected");

    const auto empty = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", ""}}, true);
    check(!empty.ok, "empty max_frames is rejected");

    const auto overflow = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"max_frames", "99999999999"}}, true);
    check(!overflow.ok, "max_frames beyond int32 range is rejected");
}

static void test_ignores_unknown_params() {
    // Unknown keys are backend-specific knobs meant for other TTS engines. A
    // request routed here must not fail just because it carries them.
    const auto opts = llama_grpc::parse_tts_request_options(
        "Hello world", "/models/voices/ref.wav", "", {{"exaggeration", "0.7"}}, true);

    check(opts.ok, "unknown params are ignored, not rejected");
}

int main() {
    test_accepts_a_minimal_valid_request();
    test_rejects_empty_text();
    test_rejects_missing_speaker_reference();
    test_accepts_named_voice_without_reference();
    test_accepts_empty_voice_without_reference();
    test_a_model_that_clones_and_describes_takes_either_or_neither();
    test_a_required_reference_keeps_its_rules_and_gains_the_transcript();
    test_parses_sampling_params();
    test_rejects_malformed_sampling_params();
    test_rejects_out_of_range_sampling_params();
    test_sampling_validation_applies_without_reference();
    test_accepts_sampling_param_boundaries();
    test_parses_max_frames();
    test_rejects_malformed_max_frames();
    test_parses_the_decoding_s_seed_steps_and_guidance();
    test_rejects_malformed_decoding_params();
    test_ignores_unknown_params();

    if (failures == 0) {
        std::printf("tts_request_options_test: all checks passed\n");
    }
    return failures;
}
