from otter.test_files import test_case

OK_FORMAT = False

name = "q3b"
points = 2

@test_case(points=None, hidden=False)
def test_q3b_matches_pytorch(scaled_dot_product_attention):
    import torch
    import torch.nn.functional as F
    Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    K = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    V = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    got = scaled_dot_product_attention(Q, K, V, mask=None)
    expected = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
    assert tuple(got.shape) == tuple(expected.shape)
    assert torch.allclose(got, expected, atol=1e-05), 'scaled_dot_product_attention should match torch.nn.functional.scaled_dot_product_attention'

