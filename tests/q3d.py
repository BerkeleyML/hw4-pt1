from otter.test_files import test_case

OK_FORMAT = False

name = "q3d"
points = 2

@test_case(points=None, hidden=False)
def test_q3d_forward_public1(MultiHeadAttention):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q3d_forward_public1.pt')
    model = MultiHeadAttention(num_heads=2, d_model=8)
    _require(model, ['attention_heads', 'W_out'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-05, rtol=1e-05), 'forward output values do not match the reference'

@test_case(points=None, hidden=False)
def test_q3d_forward_public2(MultiHeadAttention):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q3d_forward_public2.pt')
    model = MultiHeadAttention(num_heads=4, d_model=8)
    _require(model, ['attention_heads', 'W_out'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-05, rtol=1e-05), 'forward output values do not match the reference'

