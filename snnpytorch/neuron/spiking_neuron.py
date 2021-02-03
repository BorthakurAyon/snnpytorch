"""
Spiking Neuron Model
================================================
"""

import torch.nn as nn
import torch


class SpikingNeuronLayer(nn.Module):
    """
    Class of a spiking neuron layer.

    The iterative model for a neuron is inspired from the following papers:\n
    'Enabling Deep Spiking Neural Networks with Hybrid Conversion and
    Spike Timing Dependent Backpropagation' by Rathi et al. , ICLR 2019,
    https://openreview.net/forum?id=B1xSperKvH \n
    'Spatio-Temporal Backpropagation for Training High-Performance
    Spiking Neural Networks' by Wu et al.,
    https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full \n

    In PyTorch, the leaky neuron membrane voltage (u) is updated in a
    discrete manner. The membrane update at an iteration is the sum of
    input current (i) and a decaying previous membrane voltage. From this
    membrane voltage, spiking threshold (v) of spiked neurons are subtracted.
    Mathematically,

    .. math::
        u^{t} = \lambda u^{t-1} + i - vo^{t-1}\n
        o^{t-1} = 1 \: if \: u^{t-1} \:> v\: else \:0\n
    where :math:`\\lambda` is the membrane time constant,
    t is the present timestep/iteration.

    """
    def __init__(self, num_neurons=100,
                 spiking_threshold=1,
                 membrane_potential_decay_factor=0.1):
        """

        :param num_neurons: Size of neuron layer
        :param spiking_threshold: Spiking threshold voltage of neurons
        :param membrane_potential_decay_factor: Membrane time constant
        """

        super().__init__()

        # Initialize neuron layer constants
        self.num_neurons = num_neurons
        self.spiking_threshold = torch.tensor(spiking_threshold)
        self.membrane_potential_decay_factor = membrane_potential_decay_factor

    def initialize_states(self, layer_shape=None,
                          model_device='cuda:0') -> None:
        """

        Initialize the layer variable parameters: membrane potential and
        neuron spiked.\n

        :param layer_shape: ( batch size, num_neurons )
        :param model_device: 'cpu' or 'cuda:0'
        """

        if layer_shape is None:
            layer_shape = (1, self.num_neurons)

        # Declare Variable parameters
        self.register_buffer("volt_mem", torch.zeros(layer_shape,
                                                     dtype=torch.float,
                                                     device=model_device))
        self.register_buffer("neurons_spiked", torch.zeros(layer_shape,
                                                           dtype=torch.float,
                                                           device=model_device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Forward pass for this spiking neuron layer

        :param x: Synaptic current input
        :return: Binary tensor, 1 for spiked neurons, 0 for non-spiked neurons
        """

        # Check for variable dimensions and device used
        if not hasattr(self, 'volt_mem'):
            self.initialize_states(layer_shape=x.shape, model_device=x.device)

        # Check for matching of batch sizes
        # Shape check is redundant as batch size is always 1
        try:
            assert x.shape == self.volt_mem.shape
            assert x.device == self.volt_mem.device
        except AssertionError as e:
            self.initialize_states(layer_shape=x.shape, model_device=x.device)

        # Update membrane potential of neurons
        self.compute_membrane_potentials(x)
        x = self.check_for_spikes()
        return x

    def compute_membrane_potentials(self, x: torch.Tensor) -> None:
        """

        Update neuron membrane potentials.\n
        :param x: Input Synaptic current
        """

        _prev_volt = self.membrane_potential_decay_factor * self.volt_mem
        _reset = self.neurons_spiked * self.spiking_threshold
        self.volt_mem = _prev_volt + x - _reset

    def check_for_spikes(self) -> torch.Tensor:
        """

        Check for membrane voltages which exceed the spiking threshold.\n
        :return: Binary tensor, 1 for spiked neurons, 0 for non-spiked neurons
        """

        # Apply relu on the membrane voltage
        self.neurons_spiked = nn.functional.relu(self.volt_mem)

        # Divide relu outputs values by spiking threshold
        self.neurons_spiked = torch.div(self.neurons_spiked,
                                        self.spiking_threshold)

        # Create binary tensor. Values less than threshold are set to 0 and
        # values greater than threshold are set to 1.
        self.neurons_spiked = torch.floor(self.neurons_spiked)
        return self.neurons_spiked
