package voiceprofile

// AbandonForTest ends a Store the way a killed process ends: the kernel drops
// its lock on the owner file and nothing is removed.
func (s *Store) AbandonForTest() {
	if s.leaseOwner != nil {
		_ = s.leaseOwner.Close()
		s.leaseOwner = nil
	}
}

const (
	LeasesDirNameForTest      = leasesDirName
	LeaseOwnerFileNameForTest = leaseOwnerFileName
	AbandonedLeaseAgeForTest  = abandonedLeaseAge
)
