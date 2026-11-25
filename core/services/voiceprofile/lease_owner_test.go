package voiceprofile_test

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/mudler/LocalAI/core/services/voiceprofile"
)

// recordings lists every WAV file below the store, leases included.
func recordings(dataPath string) []string {
	var found []string
	root := filepath.Join(dataPath, voiceprofile.DirectoryName)
	Expect(filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".wav") {
			rel, _ := filepath.Rel(root, path)
			found = append(found, rel)
		}
		return nil
	})).To(Succeed())
	return found
}

var _ = Describe("Request leases of a store that is gone", func() {
	var dataPath string

	newStore := func() *voiceprofile.Store {
		store := voiceprofile.NewStore(dataPath)
		DeferCleanup(func() { Expect(store.Close()).To(Succeed()) })
		return store
	}
	create := func(ctx SpecContext, store *voiceprofile.Store, name string) voiceprofile.Profile {
		created, err := store.Create(ctx, voiceprofile.CreateInput{
			Name: name, Transcript: "A reference sentence.", ConsentConfirmed: true,
		}, bytes.NewReader(pcmWAV(2*time.Second)))
		Expect(err).NotTo(HaveOccurred())
		return created
	}

	BeforeEach(func() {
		dataPath = GinkgoT().TempDir()
	})

	It("leaves nothing of a deleted recording behind when the process that leased it was killed", func(ctx SpecContext) {
		killed := newStore()
		profile := create(ctx, killed, "A person's voice")
		_, lease, _, err := killed.LeaseAudio(ctx, profile.ID)
		Expect(err).NotTo(HaveOccurred())
		Expect(lease).To(BeAnExistingFile())

		survivor := newStore()
		_, err = survivor.List(ctx)
		Expect(err).NotTo(HaveOccurred())
		// the request was never released: the process died with it
		killed.AbandonForTest()

		Expect(survivor.Delete(ctx, profile.ID)).To(Succeed())
		Expect(lease).NotTo(BeAnExistingFile())
		Expect(recordings(dataPath)).To(BeEmpty(), "a hard link under .leases keeps the recording's bytes after the delete")
		Expect(filepath.Dir(lease)).NotTo(BeADirectory())
	})

	It("keeps the lease of a living store through a delete, as a synthesis in flight needs", func(ctx SpecContext) {
		busy := newStore()
		profile := create(ctx, busy, "In flight")
		_, lease, release, err := busy.LeaseAudio(ctx, profile.ID)
		Expect(err).NotTo(HaveOccurred())

		other := newStore()
		Expect(other.Delete(ctx, profile.ID)).To(Succeed())
		Expect(lease).To(BeAnExistingFile(), "the owner is alive: its request still reads the clip")
		audio, err := os.ReadFile(lease)
		Expect(err).NotTo(HaveOccurred())
		Expect(audio).To(Equal(pcmWAV(2 * time.Second)))

		release()
		Expect(lease).NotTo(BeAnExistingFile())
		Expect(recordings(dataPath)).To(BeEmpty())
	})

	It("removes at start what the processes before it left behind", func(ctx SpecContext) {
		killed := newStore()
		profile := create(ctx, killed, "Left behind")
		_, lease, _, err := killed.LeaseAudio(ctx, profile.ID)
		Expect(err).NotTo(HaveOccurred())
		killed.AbandonForTest()

		next := newStore()
		_, err = next.List(ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(lease).NotTo(BeAnExistingFile())
		Expect(filepath.Dir(lease)).NotTo(BeADirectory())
		// the profile itself is not a lease: it stays
		Expect(recordings(dataPath)).To(ConsistOf(filepath.Join(profile.ID, "reference.wav")))
	})

	It("removes a lease directory without an owner file only once it is old", func(ctx SpecContext) {
		// two directories as a LocalAI that wrote no owner file left them
		leases := filepath.Join(dataPath, voiceprofile.DirectoryName, voiceprofile.LeasesDirNameForTest)
		old := filepath.Join(leases, "11111111-1111-4111-8111-111111111111")
		recent := filepath.Join(leases, "22222222-2222-4222-8222-222222222222")
		for _, dir := range []string{old, recent} {
			Expect(os.MkdirAll(dir, 0o700)).To(Succeed())
			Expect(os.WriteFile(filepath.Join(dir, "lease.wav"), pcmWAV(time.Second), 0o600)).To(Succeed())
		}
		past := time.Now().Add(-2 * voiceprofile.AbandonedLeaseAgeForTest)
		Expect(os.Chtimes(old, past, past)).To(Succeed())

		store := newStore()
		_, err := store.List(ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(old).NotTo(BeADirectory())
		Expect(filepath.Join(recent, "lease.wav")).To(BeAnExistingFile(), "it may belong to a process that still runs")
	})

	It("follows no link and removes no stray file under the lease directory", func(ctx SpecContext) {
		outside := GinkgoT().TempDir()
		kept := filepath.Join(outside, "not-a-lease.wav")
		Expect(os.WriteFile(kept, pcmWAV(time.Second), 0o600)).To(Succeed())
		past := time.Now().Add(-2 * voiceprofile.AbandonedLeaseAgeForTest)
		Expect(os.Chtimes(outside, past, past)).To(Succeed())

		leases := filepath.Join(dataPath, voiceprofile.DirectoryName, voiceprofile.LeasesDirNameForTest)
		Expect(os.MkdirAll(leases, 0o700)).To(Succeed())
		Expect(os.Symlink(outside, filepath.Join(leases, "33333333-3333-4333-8333-333333333333"))).To(Succeed())
		stray := filepath.Join(leases, "stray-file")
		Expect(os.WriteFile(stray, []byte("x"), 0o600)).To(Succeed())
		Expect(os.Chtimes(stray, past, past)).To(Succeed())

		store := newStore()
		_, err := store.List(ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(kept).To(BeAnExistingFile())
		Expect(stray).To(BeAnExistingFile())
	})

	It("locks an owner file while it lives and leaves none when it closes", func(ctx SpecContext) {
		store := voiceprofile.NewStore(dataPath)
		profile := create(ctx, store, "Owner")
		_, lease, release, err := store.LeaseAudio(ctx, profile.ID)
		Expect(err).NotTo(HaveOccurred())
		owner := filepath.Join(filepath.Dir(lease), voiceprofile.LeaseOwnerFileNameForTest)
		Expect(owner).To(BeAnExistingFile())
		info, err := os.Stat(owner)
		Expect(err).NotTo(HaveOccurred())
		Expect(info.Mode().Perm()).To(Equal(os.FileMode(0o600)))
		release()

		Expect(store.Close()).To(Succeed())
		Expect(filepath.Dir(lease)).NotTo(BeADirectory())
	})
})
