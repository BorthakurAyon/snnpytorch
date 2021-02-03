from snnpytorch.network.spiking_neural_network import SNN
import torch


def test_basic_pipeline():
    """

    :return:
    """
    model = SNN()
    x = torch.zeros((1, 10), device='cpu')
    x[0, [2, 4, 5, 6]] = 1
    output_spikes = model.forward(x)

    assert output_spikes.shape[1] != 0
