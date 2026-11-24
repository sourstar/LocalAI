// What the llama.cpp gRPC backend does on a SECOND termination signal.
//
// The first SIGTERM/SIGINT ends the server loop and main() cleans up: the gRPC
// server's shutdown, the GPU runtime's teardown. A second one, arriving while
// that runs, used to call exit(1) from inside the signal handler. exit() runs
// the exit handlers and the static destructors - on a thread that interrupted,
// or runs beside, the very clean-up those destructors belong to - and can
// block there for good: "terminating immediately" printed, the process still
// alive, its GPU memory still held, until a supervisor's SIGKILL.
//
// A backend gets two signals whenever something between LocalAI and it
// forwards one: LocalAI stops a backend with SIGTERM to its process GROUP, so
// a wrapper script or supervisor in that group that passes the signal on to
// its child delivers a second one a millisecond after the first.
//
// terminate_on_second_signal() leaves at once with async-signal-safe calls
// only (POSIX: write(2) and _Exit; no stdio, no exit handlers, no destructors).
// The operating system reclaims the process's memory, the GPU's included.
//
// Dependency-free (C and C++ standard library, unistd.h) so that
// second_signal_test.cpp can exercise it without building the backend.
#ifndef LLAMA_GRPC_SECOND_SIGNAL_H
#define LLAMA_GRPC_SECOND_SIGNAL_H

#include <cstdlib>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace llama_grpc {

[[noreturn]] inline void terminate_on_second_signal() {
    static const char msg[] = "Received second interrupt, terminating immediately.\n";
#if defined(_WIN32)
    (void) _write(2, msg, (unsigned int) (sizeof(msg) - 1));
#else
    const ssize_t written = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void) written;
#endif
    std::_Exit(1);
}

} // namespace llama_grpc

#endif // LLAMA_GRPC_SECOND_SIGNAL_H
