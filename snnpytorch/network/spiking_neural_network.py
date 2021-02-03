"""
Spiking Neural Network
========================
"""

from snnpytorch.neuron.spiking_neuron import SpikingNeuronLayer
from snnpytorch.synapse.exc_synapse import SynapseLayer
import torch.nn as nn
import torch


class SNN(nn.Module):
    """
    Class of a spiking neural network
    """
    def __init__(self, num_input_neurons=10,
                 num_output_neurons=100,
                 conn_prob=0.5):
        """
        :param num_input_neurons: Number of neurons in input layer
        :param num_output_neurons: Number of neurons in output layer
        :param conn_prob: Connection probability from input to output layer
        """
        super().__init__()

        # Create input Synapse layer
        self.input_synapse = SynapseLayer(num_input_neurons=num_input_neurons,
                                          num_output_neurons=num_output_neurons,
                                          conn_prob=conn_prob)

        # Create output Neuron layer
        self.layer1 = SpikingNeuronLayer(num_neurons=num_output_neurons)

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for this spiking neuron network

        :param input_spikes: Network input data of spike raster
        :return: Network output data of spike raster
        """

        # Compute synaptic input for spiking neuron layer
        x = self.input_synapse(input_spikes)

        # Compute output spikes
        x = self.layer1(x)
        return x
