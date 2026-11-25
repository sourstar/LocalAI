//go:build windows

package voiceprofile

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// tryLockLeaseOwner takes an exclusive, non-blocking lock on f using Windows
// LockFileEx. It reports false, without an error, when another handle holds
// the lock.
func tryLockLeaseOwner(f *os.File) (bool, error) {
	err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, new(windows.Overlapped))
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return false, nil
	}
	return err == nil, err
}
