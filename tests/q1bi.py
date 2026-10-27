from otter.test_files import test_case

OK_FORMAT = False

name = "q1bi"
points = 1

@test_case(points=None, hidden=False)
def test_q1bi_setup(cnn_train_accuracies, cnn_train_losses, cnn_val_accuracies, cnn_val_losses):
    histories = [cnn_train_accuracies, cnn_train_losses, cnn_val_accuracies, cnn_val_losses]
    assert all((isinstance(history, list) and len(history) == 20 for history in histories))

