"""
Scaffolding only — the temporal branch itself is intentionally not built
yet (see tasks/IMPROVEMENT_PLAN.md, Phase 4, and temporal/README.md). This
defines the aggregator module ahead of time so the interface it needs from
the spatial/frequency branches — a per-frame embedding, not a per-frame
prediction — is decided before those branches get wired into it.

TemporalAttentionPool takes a sequence of per-frame embeddings for one
video and learns to weight frames by how much evidence they carry, instead
of predict_video.py's current np.mean() over per-frame probabilities.
"""
import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, frame_embeddings):
        """
        frame_embeddings: (batch, num_frames, embed_dim)
        """
        weights = torch.softmax(self.attention(frame_embeddings), dim=1)  # (B, T, 1)
        pooled = (weights * frame_embeddings).sum(dim=1)  # (B, embed_dim)
        return self.classifier(pooled)


def demo():
    """Smallest possible check that the module is wired up correctly."""
    batch, num_frames, embed_dim = 4, 16, 512

    model = TemporalAttentionPool(embed_dim=embed_dim)
    dummy = torch.randn(batch, num_frames, embed_dim)

    logits = model(dummy)
    assert logits.shape == (batch, 2), logits.shape

    print("TemporalAttentionPool smoke test passed:", tuple(logits.shape))


if __name__ == "__main__":
    demo()
