"""

Test and Plot Spike Raster
===========================
"""

from snnpytorch.network.spiking_neural_network import SNN
from snnpytorch.dataset.spike_raster import SpikeRaster
from time import time as t
from torch.utils import data
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import os


class Pipeline(object):
    """

    Class for testing the network and plotting spike raster
    """

    def __init__(self):
        self.num_steps = None
        self.num_input_neurons = None
        self.num_output_neurons = None
        self.conn_prob = None
        self.device = None

    def parse_cmd(self) -> None:
        """

        Parse command line inputs and update the model
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--num_steps", type=int, default=200)
        parser.add_argument("--num_input_neurons", type=int, default=10)
        parser.add_argument("--num_output_neurons", type=int, default=100)
        parser.add_argument("--conn_prob", type=float, default=0.5)
        args = parser.parse_args()

        self.num_steps = args.num_steps
        self.num_input_neurons = args.num_input_neurons
        self.num_output_neurons = args.num_output_neurons
        self.conn_prob = args.conn_prob
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def gen_input_spike_raster(self) -> None:
        """

        Generate and save input spike raster.
        """

        input_data = np.zeros((self.num_steps, self.num_input_neurons))
        expectation = 5
        num_spikes_per_neuron = int(self.num_steps / expectation)

        # Make the spike data
        for input_neuron in range(self.num_input_neurons):
            isi = np.random.poisson(expectation, num_spikes_per_neuron)
            spike_times = np.cumsum(isi)
            input_data[
                spike_times[spike_times < self.num_steps], input_neuron] = 1

        # Save the data for later use by dataloader
        file_name = Path.cwd() / 'snnpytorch' / 'data' / 'input_data.pt'
        torch.save(input_data, file_name)

    def load_dataset(self) -> torch.Tensor:
        """

        Load spike data for time points using torch dataloader\n
        :return: Input spike data for a single time point
        """

        # Parameters
        params = {'shuffle': False,
                  'batch_size': 1
                  }

        file_name = Path.cwd() / 'snnpytorch' / 'data' / 'input_data.pt'

        # Generators
        input_data = SpikeRaster(fname=file_name)
        input_data_generator = data.DataLoader(input_data, **params)
        return input_data_generator

    def run(self) -> (torch.Tensor, torch.Tensor):
        """

        Run the simulation for the defined number of steps.\n
        :return: input spike raster, output spike raster
        """

        output_spikes = []
        input_spikes = []

        # Update parameters as specified from command line
        self.parse_cmd()

        # Generate input spike raster ( self.num_steps, self.num_input_neurons )
        self.gen_input_spike_raster()

        # Load dataset
        input_data_generator = self.load_dataset()

        # Create spiking neuron model
        model = SNN(num_input_neurons=self.num_input_neurons,
                    num_output_neurons=self.num_output_neurons,
                    conn_prob=self.conn_prob)
        model.to(self.device)

        start = t()
        print("\nProgress: ""(%.4f seconds)" % (t() - start))

        progress = tqdm(total=len(input_data_generator))

        for local_data in input_data_generator:
            # Transfer data to GPU / CPU
            local_data = local_data.to(self.device).float()

            with torch.no_grad():  # No learning in the model
                input_spikes.append(local_data.tolist()[0])  # batch size=1

                # Update model with the data and store the spike outputs
                output_spikes.append(model.forward(local_data).tolist()[0])
                progress.update(1)

        progress.close()
        return torch.tensor(input_spikes), torch.tensor(output_spikes)

    def plot_simulation_results(self, input_spike_data: torch.Tensor,
                                output_spike_data: torch.Tensor) -> None:
        """

        Plot input and output spike raster.\n
        :param input_spike_data: Input spike raster
        :param output_spike_data: Output spike raster
        """

        fig, ax = plt.subplots(1, 2)

        # Input spike raster plot
        ax[0].scatter(*torch.where(input_spike_data), color='k')
        ax[0].set_xlim([0, self.num_steps])
        ax[0].set_ylabel(" Number of input channels (M) ")
        ax[0].set_xlabel(" Number of time points (T) ")
        ax[0].set_title(" Spike raster plot ")

        # Output spike raster plot
        ax[1].scatter(*torch.where(output_spike_data), color='r')
        ax[1].set_xlim([0, self.num_steps])
        ax[1].set_xlabel(" Number of time points (T) ")
        ax[1].set_ylabel(" Number of output neurons (N) ")
        ax[1].set_title(" Spike raster plot ")

        # plot if display is available
        have_display = bool(os.environ.get('DISPLAY', None))
        if have_display:
            plt.show()
            filename = Path.cwd() / 'snnpytorch' / 'data' / \
                'input_output_spike_raster.png'
            plt.savefig(filename)
        else:
            filename = Path.cwd() / 'snnpytorch' / 'data' / \
                'input_output_spike_raster.png'
            plt.savefig(filename)


if __name__ == "__main__":
    pipeline = Pipeline()
    input_spikes, output_spikes = pipeline.run()

    # Plot raster if device display is available
    pipeline.plot_simulation_results(input_spikes, output_spikes)
    print("End of simulation")
