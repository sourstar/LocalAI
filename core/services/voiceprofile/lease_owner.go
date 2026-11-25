package voiceprofile

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/mudler/xlog"
)

const (
	// leasesDirName holds one directory of request leases per Store.
	leasesDirName = ".leases"

	// leaseOwnerFileName is the file a Store keeps locked in its lease
	// directory for as long as it lives. A lease directory whose owner file
	// can be locked by someone else belongs to a process that is gone.
	leaseOwnerFileName = ".owner"

	// abandonedLeaseAge is how long a lease directory WITHOUT an owner file
	// (made by a LocalAI that did not write one) is left alone, counted from
	// the last lease made or released in it. No synthesis holds a lease that
	// long.
	abandonedLeaseAge = 24 * time.Hour
)

// claimLeaseDirectory writes and locks this Store's owner file. The lock is
// held until Close, or until the process ends, whichever comes first - which
// is the point: the kernel drops it when the process dies, however it dies.
func (s *Store) claimLeaseDirectory(root *os.Root) error {
	owner, err := root.OpenFile(filepath.Join(s.leaseDir, leaseOwnerFileName), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	if locked, err := tryLockLeaseOwner(owner); err != nil || !locked {
		_ = owner.Close()
		if err == nil {
			err = errors.New("lease owner file is locked by another process")
		}
		return err
	}
	s.leaseOwner = owner
	return nil
}

// purgeAbandonedLeases removes the lease directories of Stores that are gone,
// with every lease in them.
//
// A lease is a hard link to a profile's reference clip, made for one request
// and removed when the request ends. A process that is killed removes
// nothing: its directory stays, and a lease in it keeps the recording's bytes
// on disk after the profile is deleted - the one thing a delete of a person's
// voice must not leave behind. A living Store's leases are never touched: a
// synthesis in flight keeps its audio until it completes.
func (s *Store) purgeAbandonedLeases(root *os.Root) {
	leases, err := root.Open(leasesDirName)
	if err != nil {
		return
	}
	entries, err := leases.ReadDir(-1)
	_ = leases.Close()
	if err != nil {
		xlog.Warn("Failed to list voice profile lease directories", "error", err)
		return
	}
	own := filepath.Base(s.leaseDir)
	for _, entry := range entries {
		// only real directories: a link is not followed, a stray file is not ours
		if !entry.Type().IsDir() || entry.Name() == own {
			continue
		}
		dir := filepath.Join(leasesDirName, entry.Name())
		if !s.leaseDirectoryAbandoned(root, dir) {
			continue
		}
		if err := root.RemoveAll(dir); err != nil && !errors.Is(err, fs.ErrNotExist) {
			xlog.Warn("Failed to remove an abandoned voice profile lease directory", "error", err)
			continue
		}
		xlog.Info("Removed the voice profile leases of a process that is gone", "directory", entry.Name())
	}
}

// leaseDirectoryAbandoned reports whether the Store that made dir is gone.
func (s *Store) leaseDirectoryAbandoned(root *os.Root, dir string) bool {
	owner, err := root.OpenFile(filepath.Join(dir, leaseOwnerFileName), os.O_RDWR, 0)
	if errors.Is(err, fs.ErrNotExist) {
		// no owner file: a directory of a LocalAI that wrote none. Its age is
		// the directory's own time - a lease is a link, so a lease's time is
		// its clip's, not the request's.
		info, statErr := root.Stat(dir)
		return statErr == nil && s.now().Sub(info.ModTime()) > abandonedLeaseAge
	}
	if err != nil {
		return false
	}
	defer func() { _ = owner.Close() }()
	locked, err := tryLockLeaseOwner(owner)
	// the lock is ours until owner is closed: the directory is removed under it
	return err == nil && locked
}
