from otter.test_files import test_case

OK_FORMAT = False

name = "q2ci"
points = 1

@test_case(points=None, hidden=False)
def test_q2ci_setup(resnet_train_accuracies, resnet_train_losses, resnet_val_accuracies, resnet_val_losses):
    histories = [resnet_train_accuracies, resnet_train_losses, resnet_val_accuracies, resnet_val_losses]
    assert all((isinstance(history, list) and len(history) == 50 for history in histories))

