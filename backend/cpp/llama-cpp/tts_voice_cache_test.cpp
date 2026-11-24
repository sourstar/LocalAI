// Unit tests for the place of a voice clip's kept encoding (tts_voice_cache.h).
//
// Build & run standalone (standard library only):
//   g++ -std=c++17 tts_voice_cache_test.cpp -o t && ./t
//
// Builds LocalAI's voice-profile store in a temporary directory - a profile's clip, its lease as a hard link - and
// checks that the encoding is placed beside the profile's own clip, and nowhere when the clip is not such a lease.
// On Windows this file compiles to a no-op success (no file is ever named there).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "tts_voice_cache.h"

static int failures = 0;

static void check(bool ok, const std::string & name) {
    if (!ok) {
        failures++;
        fprintf(stderr, "FAIL: %s\n", name.c_str());
    } else {
        fprintf(stderr, "ok:   %s\n", name.c_str());
    }
}

#if !defined(_WIN32)
#include <unistd.h>

namespace fs = std::filesystem;

static void write_file(const fs::path & p, const std::string & bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << bytes;
}

static void test_place() {
    using llama_grpc::voice_clip_cache_path;
    char tmpl[] = "/tmp/tts-voice-cache-XXXXXX";
    if (mkdtemp(tmpl) == nullptr) {
        check(false, "a temporary directory");
        return;
    }
    const fs::path store = fs::path(tmpl) / "voice-profiles";

    // two profiles, one with two references; a lease of each clip, as LeaseAudios makes them
    write_file(store / "anna" / "reference.wav", "the first voice");
    write_file(store / "anna" / "profile.json", "{}");
    write_file(store / "bruno" / "reference.wav", "the second voice");
    write_file(store / "bruno" / "reference-1.wav", "the second voice, again");
    const fs::path leases = store / ".leases" / "process-1";
    fs::create_directories(leases);
    const fs::path lease_a  = leases / "request-1.wav";
    const fs::path lease_b1 = leases / "request-2.wav";
    fs::create_hard_link(store / "anna" / "reference.wav", lease_a);
    fs::create_hard_link(store / "bruno" / "reference-1.wav", lease_b1);

    check(voice_clip_cache_path(lease_a.string()) == (store / "anna" / "reference.wav.encoded").string(),
          "a lease leads to a file beside the profile's own clip");
    check(voice_clip_cache_path(lease_b1.string()) == (store / "bruno" / "reference-1.wav.encoded").string(),
          "each reference of a profile has its own");

    // same bytes, another profile: found by the file, not by its content
    write_file(store / "carla" / "reference.wav", "the first voice");
    check(voice_clip_cache_path(lease_a.string()) == (store / "anna" / "reference.wav.encoded").string(),
          "a profile with the same bytes is another profile");

    // a lease that is a copy (LocalAI's fallback where hard links fail): no other name leads to it
    const fs::path copy = leases / "request-3.wav";
    fs::copy_file(store / "anna" / "reference.wav", copy);
    check(voice_clip_cache_path(copy.string()).empty(), "a lease that is a copy gets no file");

    // not a lease at all: a clip somewhere else, a missing file, a directory, an empty path
    write_file(fs::path(tmpl) / "elsewhere" / "x" / "clip.wav", "a clip of the caller's");
    check(voice_clip_cache_path((fs::path(tmpl) / "elsewhere" / "x" / "clip.wav").string()).empty(), "a clip that is no lease gets no file");
    check(voice_clip_cache_path((leases / "missing.wav").string()).empty(), "nor a lease that is gone");
    check(voice_clip_cache_path(leases.string()).empty(), "nor a directory");
    check(voice_clip_cache_path("").empty() && voice_clip_cache_path("clip.wav").empty(), "nor a path with no store above it");

    // the profile deleted while a request still holds its lease (LocalAI's RemoveAll): nothing is named any more
    fs::remove_all(store / "anna");
    check(voice_clip_cache_path(lease_a.string()).empty(), "a deleted profile's lease gets no file");

    // a link to a profile directory is not a profile
    fs::create_directory_symlink(store / "bruno", store / "alias");
    check(voice_clip_cache_path(lease_b1.string()) == (store / "bruno" / "reference-1.wav.encoded").string(),
          "a link to a profile is not followed");

    // an engine that takes the file is given it, one that does not is left alone
    struct takes { std::string got; void set_speaker_ref_cache(std::string p) { got = std::move(p); } };
    struct takes_not { };
    takes with;
    takes_not without;
    check(llama_grpc::set_speaker_ref_cache(with, lease_b1.string()) && with.got == (store / "bruno" / "reference-1.wav.encoded").string(),
          "an engine that keeps encodings is told the file");
    check(!llama_grpc::set_speaker_ref_cache(without, lease_b1.string()), "an engine that does not is not");
    takes none;
    check(!llama_grpc::set_speaker_ref_cache(none, copy.string()) && none.got.empty(), "no file, nothing told");

    fs::remove_all(tmpl);
}
#else
static void test_place() {}
#endif

int main() {
    test_place();
    if (failures > 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    fprintf(stderr, "all tests passed\n");
    return 0;
}
