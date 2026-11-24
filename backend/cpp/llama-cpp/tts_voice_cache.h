// Where the engine may keep its encoding of a voice profile's reference clip.
//
// A model that clones a voice encodes the reference clip first (OmniVoice: about a second on a GPU). The engine keeps
// the encoding in memory between requests; a model that has just been loaded encodes again. Given the path of ONE
// file, the engine keeps the encoding there and a later load reads it (mtmd_helper_gen_audio_inp::speaker_ref_cache).
//
// The encoding is the voice, as much as the clip is, so that file must live and die with the profile. LocalAI hands
// a profile's clip to a backend as a request-scoped HARD LINK,
//     <data>/voice-profiles/.leases/<process>/<request>.wav  ->  <data>/voice-profiles/<id>/reference*.wav
// (core/services/voiceprofile, LeaseAudios), and deletes a profile with RemoveAll(<id>). voice_clip_cache_path()
// finds the profile's own file by the lease's inode and names a file beside it: inside the profile's directory, with
// its permissions, removed by the same RemoveAll. Anything else - a clip that is no lease, a lease that is a copy
// (no hard link on that filesystem), a store laid out otherwise - gets no file: the encoding stays in memory, as
// before.
//
// Dependency-free (C++17 standard library, POSIX stat) so that tts_voice_cache_test.cpp tests it without the backend.
#ifndef LLAMA_GRPC_TTS_VOICE_CACHE_H
#define LLAMA_GRPC_TTS_VOICE_CACHE_H

#include <filesystem>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace llama_grpc {

// the suffix of the kept encoding beside a profile's clip
inline constexpr const char * VOICE_CLIP_CACHE_SUFFIX = ".encoded";

// the file beside the profile's own clip in which the engine may keep the encoding of `clip_path`, "" if there is none
inline std::string voice_clip_cache_path(const std::string & clip_path) {
#if defined(_WIN32)
    (void) clip_path;
    return {};
#else
    namespace fs = std::filesystem;
    const fs::path lease(clip_path);
    const fs::path leases = lease.parent_path().parent_path(); // <store>/.leases
    if (leases.filename() != ".leases" || !leases.has_parent_path()) {
        return {};
    }
    struct stat want;
    if (stat(clip_path.c_str(), &want) != 0 || !S_ISREG(want.st_mode) || want.st_nlink < 2) {
        return {}; // not there, or a copy: no other name leads to these bytes
    }
    std::error_code ec;
    for (fs::directory_iterator profiles(leases.parent_path(), ec), end; !ec && profiles != end; profiles.increment(ec)) {
        const std::string name = profiles->path().filename().string();
        struct stat dir;
        if (name.empty() || name[0] == '.' || lstat(profiles->path().c_str(), &dir) != 0 || !S_ISDIR(dir.st_mode)) {
            continue; // the leases, temporary directories, links
        }
        std::error_code ec_files;
        for (fs::directory_iterator files(profiles->path(), ec_files); !ec_files && files != end; files.increment(ec_files)) {
            struct stat have;
            if (lstat(files->path().c_str(), &have) == 0 && S_ISREG(have.st_mode) &&
                have.st_dev == want.st_dev && have.st_ino == want.st_ino) {
                return files->path().string() + VOICE_CLIP_CACHE_SUFFIX;
            }
        }
    }
    return {};
#endif
}

// A llama.cpp whose speech input takes the file (set_speaker_ref_cache) is given it; one that does not, is not: the
// backend builds against both, and without the method the encoding stays in memory.
template <typename Inp, typename = void>
struct has_speaker_ref_cache : std::false_type {};
template <typename Inp>
struct has_speaker_ref_cache<Inp, std::void_t<decltype(std::declval<Inp &>().set_speaker_ref_cache(std::string()))>> : std::true_type {};

// names the file of clip_path's encoding to the engine, if there is one and the engine takes it; says whether it did
template <typename Inp>
bool set_speaker_ref_cache(Inp & inp, const std::string & clip_path) {
    if constexpr (has_speaker_ref_cache<Inp>::value) {
        const std::string kept = voice_clip_cache_path(clip_path);
        if (!kept.empty()) {
            inp.set_speaker_ref_cache(kept);
            return true;
        }
    } else {
        (void) inp;
        (void) clip_path;
    }
    return false;
}

} // namespace llama_grpc

#endif // LLAMA_GRPC_TTS_VOICE_CACHE_H
