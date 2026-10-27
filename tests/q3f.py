from otter.test_files import test_case

OK_FORMAT = False

name = "q3f"
points = 3

@test_case(points=None, hidden=False)
def test_q3f_forward_public1(PositionalEncoding):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q3f_forward_public1.pt')
    model = PositionalEncoding(8, max_len=16)
    _require(model, ['pos'])
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-05, rtol=1e-05), 'forward output values do not match the reference'

@test_case(points=None, hidden=False)
def test_q3f_forward_public2(PositionalEncoding):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q3f_forward_public2.pt')
    model = PositionalEncoding(8, max_len=16)
    _require(model, ['pos'])
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-05, rtol=1e-05), 'forward output values do not match the reference'

