from otter.test_files import test_case

OK_FORMAT = False

name = "q2a"
points = 2

@test_case(points=None, hidden=False)
def test_q2a_forward_public1(ResidualBlock):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q2a_forward_public1.pt')
    model = ResidualBlock(in_channels=3, out_channels=8, initial_downsample=False)
    _require(model, ['conv1', 'batchnorm1', 'conv2', 'batchnorm2', 'residual_connection'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-05, rtol=1e-05), 'forward output values do not match the reference'

@test_case(points=None, hidden=False)
def test_q2a_forward_public2(ResidualBlock):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q2a_forward_public2.pt')
    model = ResidualBlock(in_channels=4, out_channels=8, initial_downsample=True)
    _require(model, ['conv1', 'batchnorm1', 'conv2', 'batchnorm2', 'residual_connection'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-05, rtol=1e-05), 'forward output values do not match the reference'

