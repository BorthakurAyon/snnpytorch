"""
Excitatory Synapse
========================================
"""

import torch.nn as nn
import numpy as np
import torch


class SynapseLayer(nn.Module):
    """

    Class of a synapse layer.

    The iterative model for the synapse is inspired from the following papers:\n
    'Enabling Deep Spiking Neural Networks with Hybrid Conversion and
    Spike Timing Dependent Backpropagation' by Rathi et al. , ICLR 2019,
    https://openreview.net/forum?id=B1xSperKvH \n
    'Spatio-Temporal Backpropagation for Training High-Performance
    Spiking Neural Networks' by Wu et al.,
    https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full \n

    In PyTorch, the total synaptic input current to a neuron / synapse states
    (i) update is performed in the following manner:

    .. math::
        i^{t} = \lambda i^{t-1} + wI\n
    where :math:`\\lambda` is the synaptic time constant,
    t is the present timestep/iteration, w is the synaptic weight, I is the
    input spike.

    """

    def __init__(self, num_input_neurons=10, num_output_neurons=100,
                 synapse_time_constant=0.05, conn_prob=0.5,
                 initial_weight_config=None):
        """

        :param num_input_neurons: Number of neurons in input layer
        :param num_output_neurons: Number of neurons in output layer
        :param synapse_time_constant: Synapse time constant
        :param conn_prob: Connection probability from input to output layer
        :param initial_weight_config: Weight initialisation method and value.\n
        Default value of initial_weight_config
        : {"method": "constant", "value": 0.25}
        """

        super().__init__()

        if initial_weight_config is None:
            initial_weight_config = {"method": "constant", "value": 0.25}

        # Initialize synapse layer constants
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.synapse_time_constant = synapse_time_constant
        self.conn_prob = conn_prob
        self.initial_weight_config = initial_weight_config

    def initialize_states(self, model_device='cuda:0') -> None:
        """

        Initialise synapse layer weights, connectivity, and state.
        :param model_device: 'cpu' or 'cuda:0'
        """
        layer_shape = (self.num_output_neurons, self.num_input_neurons)

        # Declare Variable parameters
        self.register_buffer("weight", torch.zeros(layer_shape,
                                                   dtype=torch.float,
                                                   device=model_device))
        self.register_buffer("connectivity", torch.zeros(layer_shape,
                                                         dtype=torch.float,
                                                         device=model_device))
        self.register_buffer("synapse_state", torch.zeros((1, layer_shape[0]),
                                                          dtype=torch.float,
                                                          device=model_device))
        # Initialize weight values
        if self.initial_weight_config["method"] == 'constant':
            self.weight.fill_(self.initial_weight_config['value'])
        # Initialize connectivity tensor
        self.create_connectivity()

    def create_connectivity(self) -> None:
        """

        Create random connectivity between input and output layer neurons.
        """
        for _pre_index in range(self.num_input_neurons):
            _post_indices = np.random.choice(
                range(self.num_output_neurons),
                int(self.conn_prob * self.num_output_neurons),
                replace=False)
            self.connectivity[_post_indices, _pre_index] = \
                torch.tensor([1.], device=self.connectivity.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Forward pass for this synapse layer.
        :param x: Input neuron spikes
        :return: Synapse state / Input current to the output layer
        """

        # Check for variable dimensions and device used
        if not hasattr(self, 'weight'):
            self.initialize_states(model_device=x.device)

        x = self.update_synapse_states(x)
        return x

    def update_synapse_states(self, x: torch.Tensor):
        """

        Update the synapse state
        :param x: Input neuron spikes
        :return: Synapse state / Input current to the output layer
        """

        # Multiply previous synapse layer states with time constant.
        _prev_state = self.synapse_time_constant * self.synapse_state

        # Update synapse layer states.
        _actual_weight = torch.mul(self.weight, self.connectivity)
        _present_inp = torch.matmul(_actual_weight, x.unsqueeze(2)).squeeze(2)
        self.synapse_state = _prev_state + _present_inp
        return self.synapse_state
