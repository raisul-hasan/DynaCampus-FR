# pyrefly: ignore [missing-import]
import pytest
import torch
from src.adapter.domain_adapter import ResidualDomainAdapter

def test_domain_adapter_forward():
    adapter = ResidualDomainAdapter(feature_dim=512, hidden_dim=256)
    x = torch.randn(8, 512)
    out = adapter(x)
    assert out.shape == (8, 512)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)
