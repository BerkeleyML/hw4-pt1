from otter.test_files import test_case

OK_FORMAT = False

name = "q3hi"
points = 1

@test_case(points=None, hidden=False)
def test_q3hi_setup(transformer_train_losses, transformer_val_losses):
    assert isinstance(transformer_train_losses, list)
    assert isinstance(transformer_val_losses, list)
    assert len(transformer_train_losses) > 0
    assert len(transformer_train_losses) == len(transformer_val_losses)

