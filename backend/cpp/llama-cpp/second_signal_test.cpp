// Unit test for the second-signal exit (second_signal.h).
//
// Build & run standalone (standard library only):
//   g++ -std=c++17 second_signal_test.cpp -o t && ./t
//
// A child process registers an exit handler that never returns - the stand-in
// for a clean-up the signal interrupted - and is sent a signal whose handler
// leaves the way the backend leaves on a second signal. With
// terminate_on_second_signal() the child is gone at once, status 1. The same
// child leaving with exit(1), as the backend did, is still there a second
// later: the test shows the hang it guards against, then kills it.
//
// On Windows this file compiles to a no-op success (no fork).

#include <cstdio>
#include <cstdlib>
#include <string>

#include "second_signal.h"

static int failures = 0;

static void check(bool ok, const std::string &name) {
    if (!ok) {
        failures++;
        fprintf(stderr, "FAIL: %s\n", name.c_str());
    } else {
        fprintf(stderr, "ok:   %s\n", name.c_str());
    }
}

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

static void never_returns() {
    for (;;) {
        pause();
    }
}

static void leave_at_once(int) { llama_grpc::terminate_on_second_signal(); }
static void leave_through_exit(int) { exit(1); }

// Forks a child that leaves through `handler` on SIGTERM. Returns its wait
// status once it has ended, or -1 if it is still alive after `wait_ms` (it is
// then killed).
static int status_of_child(void (*handler)(int), long wait_ms) {
    const pid_t pid = fork();
    if (pid == 0) {
        fclose(stderr); // the message is the parent's test output otherwise
        atexit(never_returns);
        signal(SIGTERM, handler);
        raise(SIGTERM);
        for (;;) {
            pause();
        }
    }
    for (long waited = 0; waited < wait_ms; waited += 10) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) {
            return status;
        }
        const struct timespec ten_ms = {0, 10 * 1000 * 1000};
        nanosleep(&ten_ms, nullptr);
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return -1;
}

static void test_second_signal() {
    const int status = status_of_child(leave_at_once, 5000);
    check(status != -1, "the process is gone although an exit handler never returns");
    check(status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 1, "exit status 1");
    check(status_of_child(leave_through_exit, 1000) == -1,
          "exit() from the handler is still there a second later (what it replaces)");
}
#else
static void test_second_signal() {}
#endif

int main() {
    test_second_signal();
    if (failures > 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    fprintf(stderr, "all tests passed\n");
    return 0;
}
