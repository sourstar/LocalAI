//go:build !windows

package voiceprofile

import (
	"errors"
	"os"
	"syscall"
)

// tryLockLeaseOwner takes an exclusive, non-blocking flock on f. It reports
// false, without an error, when another open file holds the lock.
func tryLockLeaseOwner(f *os.File) (bool, error) {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if errors.Is(err, syscall.EWOULDBLOCK) {
		return false, nil
	}
	return err == nil, err
}
