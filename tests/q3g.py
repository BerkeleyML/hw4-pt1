from otter.test_files import test_case

OK_FORMAT = False

name = "q3g"
points = 3

@test_case(points=None, hidden=False)
def test_q3g_forward_public1(TransformerDecoder):
    import torch

    def _require(module, names):
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f'Missing required attribute(s): {missing}'

    def _load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    data = _load('test_data/public/q3g_forward_public1.pt')
    model = TransformerDecoder(vocab_size=11, d_model=8, num_layers=2, num_heads=2, max_len=16)
    _require(model, ['embedding', 'positional_encoding', 'layers', 'final_layernorm', 'linear'])
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    assert not missing, f'could not load weights; missing keys: {missing}. Use the required layer names.'
    model.eval()
    out = model(data['x'])
    expected = data['out']
    assert tuple(out.shape) == tuple(expected.shape), f'expected shape {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=0.0001, rtol=0.0001), 'forward output values do not match the reference'

@test_case(points=None, hidden=False)
def test_q3g_causal_public(TransformerDecoder):
    import torch
    torch.manual_seed(189)
    model = TransformerDecoder(vocab_size=11, d_model=8, num_layers=2, num_heads=2, max_len=16)
    model.eval()
    x = torch.randint(0, 11, (2, 6))
    x_changed = x.clone()
    x_changed[:, -1] = (x[:, -1] + 1) % 11
    with torch.no_grad():
        out = model(x)
        out_changed = model(x_changed)
    assert tuple(out.shape) == (2, 6, 11), f'expected shape (2, 6, 11), got {tuple(out.shape)}'
    assert torch.allclose(out[:, :-1], out_changed[:, :-1], atol=1e-06), 'changing the last token changed the logits at earlier positions; each position should only attend to itself and earlier positions'
    assert not torch.allclose(out[:, -1], out_changed[:, -1]), 'changing the last token should change the logits at the last position'

