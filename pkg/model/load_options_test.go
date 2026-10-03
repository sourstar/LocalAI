package model

import (
	"context"
	"net"
	"sync"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
	"github.com/mudler/LocalAI/pkg/system"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// recordingBackend answers the two RPCs a load makes and keeps what
// LoadModel received, so a test can compare it with what the loader recorded.
type recordingBackend struct {
	pb.UnimplementedBackendServer
	mu       sync.Mutex
	received *pb.ModelOptions
}

func (b *recordingBackend) Health(context.Context, *pb.HealthMessage) (*pb.Reply, error) {
	return &pb.Reply{Message: []byte("OK")}, nil
}

func (b *recordingBackend) LoadModel(_ context.Context, opts *pb.ModelOptions) (*pb.Result, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.received = proto.Clone(opts).(*pb.ModelOptions)
	return &pb.Result{Success: true}, nil
}

// A model's YAML is not what its backend runs with: LocalAI completes it
// before the load. The options the backend actually received are therefore
// recorded on the loaded model, so that they can be listed (GET /system).
var _ = Describe("Model load options", func() {
	It("records exactly what the backend's LoadModel received", func() {
		lis, err := net.Listen("tcp", "127.0.0.1:0")
		Expect(err).ToNot(HaveOccurred())
		backend := &recordingBackend{}
		server := grpc.NewServer()
		pb.RegisterBackendServer(server, backend)
		go func() { _ = server.Serve(lis) }()
		DeferCleanup(server.Stop)

		systemState, err := system.GetSystemState(system.WithModelPath(GinkgoT().TempDir()))
		Expect(err).ToNot(HaveOccurred())
		ml := NewModelLoader(systemState)
		ml.SetExternalBackend("recording", lis.Addr().String())

		_, err = ml.backendLoader(
			WithModelID("encoder"),
			WithModel("encoder.gguf"),
			WithBackendString("recording"),
			WithLoadGRPCLoadModelOpts(&pb.ModelOptions{ContextSize: 8192, NBatch: 512, Embeddings: true}),
		)
		Expect(err).ToNot(HaveOccurred())

		backend.mu.Lock()
		received := backend.received
		backend.mu.Unlock()
		Expect(received).ToNot(BeNil())
		Expect(received.ContextSize).To(Equal(int32(8192)))
		Expect(received.Model).To(Equal("encoder.gguf"), "the per-load fields are set before the call")

		loaded := ml.CheckIsLoaded("encoder")
		Expect(loaded).ToNot(BeNil())
		recorded := loaded.LoadOptions()
		Expect(proto.Equal(recorded, received)).To(BeTrue(),
			"the recorded options are the ones sent, per-load fields included")
	})

	It("keeps its own copy", func() {
		m := NewModel("encoder", "127.0.0.1:1", nil)
		Expect(m.LoadOptions()).To(BeNil(), "a model registered without a load has none")

		sent := &pb.ModelOptions{ContextSize: 4096}
		m.SetLoadOptions(sent)
		sent.ContextSize = 1
		Expect(m.LoadOptions().ContextSize).To(Equal(int32(4096)), "changing what was sent changes nothing recorded")

		got := m.LoadOptions()
		got.ContextSize = 2
		Expect(m.LoadOptions().ContextSize).To(Equal(int32(4096)), "changing what was returned changes nothing recorded")
	})
})
