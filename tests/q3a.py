from otter.test_files import test_case

OK_FORMAT = False

name = "q3a"
points = 1

@test_case(points=None, hidden=False)
def test_q3a_matches_pytorch(softmax):
    import torch
    import torch.nn.functional as F
    x = torch.tensor([1.0, 2.0, 3.0])
    got = softmax(x)
    assert torch.allclose(got, F.softmax(x, dim=-1), atol=1e-05)
    assert torch.allclose(got.sum(), torch.tensor(1.0), atol=1e-05)

