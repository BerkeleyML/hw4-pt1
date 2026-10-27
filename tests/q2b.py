from otter.test_files import test_case

OK_FORMAT = False

name = "q2b"
points = 3

@test_case(points=None, hidden=False)
def test_q2b_forward_public1(ResNet18):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q2b_forward_public1.pt')
    model = ResNet18(num_classes=3)
    _require(model, ['initial_conv', 'residual1', 'residual2', 'residual3', 'residual4', 'classification_head'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=0.0001, rtol=0.0001), 'forward output values do not match the reference'

@test_case(points=None, hidden=False)
def test_q2b_forward_public2(ResNet18):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q2b_forward_public2.pt')
    model = ResNet18(num_classes=10)
    _require(model, ['initial_conv', 'residual1', 'residual2', 'residual3', 'residual4', 'classification_head'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=0.0001, rtol=0.0001), 'forward output values do not match the reference'

