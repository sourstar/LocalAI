// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <exception>
#include <map>
#include <string>

namespace llama_grpc {

// Validated, parsed form of a backend::TTSRequest, kept free of llama.cpp,
// mtmd and gRPC headers so backend/cpp/run-unit-tests.sh can compile it as a
// standalone translation unit. grpc-server.cpp turns this into a
// mtmd_helper::gen_audio::inp.
struct tts_request_options {
    bool        ok = false;
    std::string error;

    std::string text;
    // Speaker reference file, set when the model clones a voice and the request
    // names one (Qwen3-TTS Base, which needs it; OmniVoice, which can do without).
    std::string voice_path;
    // Transcript of the speaker reference (params["ref_text"], which the voice
    // profiles fill in). Kept only beside a reference.
    std::string ref_text;
    // Named or described voice for the model's "tts" chat template (LFM3-Audio:
    // "female", "male"; OmniVoice: "female, young adult, british accent").
    // Empty means the template default.
    std::string voice;
    std::string language;

    // 0 / 0.0f mean "unset": upstream only overrides the sampler defaults when
    // the value is strictly positive.
    int32_t top_k = 0;
    float   top_p = 0.0f;

    // Upper bound on generated audio frames, exposed because the model does not
    // always emit its codec EOS and will otherwise run to the 512-frame default,
    // which is roughly 41 s at the 12.5 Hz frame rate. 0 means unset, leaving
    // that default in place.
    int32_t max_frames = 0;

    // The decoding of the audio codes, for the models that take them
    // (OmniVoice decodes in steps, from noise, under guidance): the seed of
    // the noise (the same text, voice and seed give the same audio), the
    // number of steps (fewer is faster, and another audio), the guidance
    // scale (1.0 = none). -1 / 0 / 0.0f mean "unset": a seed drawn for the
    // request, the model's own steps and guidance. A model that has no steps
    // or guidance refuses a set value itself.
    int32_t seed       = -1;
    int32_t flow_steps = 0;
    float   cfg_scale  = 0.0f;
};

namespace detail {

// Strict whole-string numeric parsing. std::stoi/stof accept trailing garbage
// ("40abc" -> 40), which would silently honour a typo'd request.
inline bool parse_whole_int32(const std::string & value, int32_t & out) {
    if (value.empty()) {
        return false;
    }
    try {
        size_t     consumed = 0;
        const long parsed   = std::stol(value, &consumed);
        if (consumed != value.size()) {
            return false;
        }
        if (parsed < INT32_MIN || parsed > INT32_MAX) {
            return false;
        }
        out = static_cast<int32_t>(parsed);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

inline bool parse_whole_float(const std::string & value, float & out) {
    if (value.empty()) {
        return false;
    }
    try {
        size_t      consumed = 0;
        const float parsed   = std::stof(value, &consumed);
        if (consumed != value.size()) {
            return false;
        }
        out = parsed;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

inline tts_request_options reject(const std::string & message) {
    tts_request_options opts;
    opts.ok    = false;
    opts.error = message;
    return opts;
}

} // namespace detail

// How the loaded model takes its voice.
enum class tts_voice_mode {
    // A name or a description for the "tts" chat template (LFM3-Audio); an
    // empty voice lets the template pick its default.
    name,
    // A speaker reference file, required: the model has no voice of its own
    // (Qwen3-TTS Base).
    reference,
    // A file to clone when the request names one, a name or a description
    // otherwise, or nothing at all: the model clones AND has a "tts" template
    // (OmniVoice).
    reference_or_name,
};

// voice_is_file is the caller's answer to "does `voice` name an existing
// file", read only in reference_or_name mode; this header does no I/O.
// instructions is the request's free-form voice description (OpenAI's
// `instructions`): it is the template's voice when the request names none.
inline tts_request_options parse_tts_request_options(
        const std::string & text,
        const std::string & voice,
        const std::string & language,
        const std::string & instructions,
        const std::map<std::string, std::string> & params,
        tts_voice_mode mode,
        bool voice_is_file) {
    if (text.empty()) {
        return detail::reject("text must be a non-empty string");
    }

    tts_request_options opts;
    opts.text     = text;
    opts.language = language;

    switch (mode) {
        case tts_voice_mode::reference:
            // The Qwen3-TTS Base checkpoints have no built-in speaker. Without a
            // reference clip the model picks an arbitrary voice, so an unset voice
            // is a request error rather than a defaulted one.
            if (voice.empty()) {
                return detail::reject("voice must name a speaker reference audio file");
            }
            opts.voice_path = voice;
            break;
        case tts_voice_mode::reference_or_name:
            if (!voice.empty() && voice_is_file) {
                opts.voice_path = voice;
            } else {
                opts.voice = voice;
            }
            break;
        case tts_voice_mode::name:
            opts.voice = voice;
            break;
    }
    if (mode != tts_voice_mode::reference && opts.voice.empty()) {
        opts.voice = instructions;
    }

    if (!opts.voice_path.empty()) {
        const auto ref_text_it = params.find("ref_text");
        if (ref_text_it != params.end()) {
            opts.ref_text = ref_text_it->second;
        }
    }

    // Both values are range-checked here rather than left to the caller: the
    // consumer copies them straight into mtmd_helper::gen_audio::inp, and only
    // its separate sampler assignment is guarded by "> 0". An out-of-range or
    // non-finite value would slip past that guard and reach llama.cpp.
    const auto top_k_it = params.find("top_k");
    if (top_k_it != params.end()) {
        if (!detail::parse_whole_int32(top_k_it->second, opts.top_k)) {
            return detail::reject("top_k must be an integer, got \"" + top_k_it->second + "\"");
        }
        if (opts.top_k < 0) {
            return detail::reject("top_k must be >= 0, got \"" + top_k_it->second + "\"");
        }
    }

    const auto top_p_it = params.find("top_p");
    if (top_p_it != params.end()) {
        if (!detail::parse_whole_float(top_p_it->second, opts.top_p)) {
            return detail::reject("top_p must be a number, got \"" + top_p_it->second + "\"");
        }
        // Phrased as a negated in-range test, not "p < 0.0f || p > 1.0f",
        // because every comparison against NaN is false: the obvious form
        // would accept NaN, and NaN then defeats the consumer's "> 0" guard
        // too, since that comparison is false as well.
        if (!(opts.top_p >= 0.0f && opts.top_p <= 1.0f)) {
            return detail::reject("top_p must be between 0.0 and 1.0, got \"" + top_p_it->second + "\"");
        }
    }

    const auto max_frames_it = params.find("max_frames");
    if (max_frames_it != params.end()) {
        if (!detail::parse_whole_int32(max_frames_it->second, opts.max_frames)) {
            return detail::reject("max_frames must be an integer, got \"" + max_frames_it->second + "\"");
        }
        if (opts.max_frames < 0) {
            return detail::reject("max_frames must be >= 0, got \"" + max_frames_it->second + "\"");
        }
    }

    // The same three fields upstream's POST /v1/audio/speech reads. -1 is its
    // spelling of "a random seed" and is taken as unset.
    const auto seed_it = params.find("seed");
    if (seed_it != params.end()) {
        if (!detail::parse_whole_int32(seed_it->second, opts.seed)) {
            return detail::reject("seed must be an integer, got \"" + seed_it->second + "\"");
        }
        if (opts.seed < -1) {
            return detail::reject("seed must be >= 0, or -1 for a random one, got \"" + seed_it->second + "\"");
        }
    }

    const auto flow_steps_it = params.find("flow_steps");
    if (flow_steps_it != params.end()) {
        if (!detail::parse_whole_int32(flow_steps_it->second, opts.flow_steps)) {
            return detail::reject("flow_steps must be an integer, got \"" + flow_steps_it->second + "\"");
        }
        // upstream's own bounds for the field
        if (opts.flow_steps < 0 || opts.flow_steps > 1024) {
            return detail::reject("flow_steps must be between 0 and 1024, got \"" + flow_steps_it->second + "\"");
        }
    }

    const auto cfg_scale_it = params.find("cfg_scale");
    if (cfg_scale_it != params.end()) {
        if (!detail::parse_whole_float(cfg_scale_it->second, opts.cfg_scale)) {
            return detail::reject("cfg_scale must be a number, got \"" + cfg_scale_it->second + "\"");
        }
        // negated, as top_p above: NaN fails every comparison
        if (!(opts.cfg_scale >= 0.0f && std::isfinite(opts.cfg_scale))) {
            return detail::reject("cfg_scale must be a finite number >= 0, got \"" + cfg_scale_it->second + "\"");
        }
    }

    opts.ok = true;
    return opts;
}

// The two modes that predate tts_voice_mode. needs_reference is the model's
// mtmd_helper_cap.audio_gen_spk_ref: a required reference, or a named voice.
inline tts_request_options parse_tts_request_options(
        const std::string & text,
        const std::string & voice,
        const std::string & language,
        const std::map<std::string, std::string> & params,
        bool needs_reference) {
    return parse_tts_request_options(text, voice, language, std::string(), params,
        needs_reference ? tts_voice_mode::reference : tts_voice_mode::name, true);
}

} // namespace llama_grpc
