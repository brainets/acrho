Folder Netwk contains data of simulation of spiiking neural networks composed of excitatory and inhibitory neurons.

The folder "homogeneous" contains data for homogeneous network (i.e. all neurons are identical).
The folder "hetrogeneous" contains data for heterogeneous networks (i.e. the reversal of inhibtory neurons is distributed according to a Normal distribution)
the notebook Read_Spike_trains.ipynb allows to read the spiking times for excitatory and inhibitory neurons.
Trial from to 0 to 99 for input A with low amplitude.
Trial from to 100 to 199 for input B with higher amplitude.

Folder spatial contains data of simulation of spatially connected mean field models of the network in Folder Netwk

The folder "homogeneous" contains data for homogeneous network (i.e. all neurons are identical).
The folder "hetrogeneous" contains data for heterogeneous networks (i.e. the reversal of inhibtory neurons is distributed according to a Normal distribution)
Files named i_jsignal.py in the folder data contains spatio-temporal profile of the system in response to a stationary stimulation.
i=0 stands for i input A with low amplitude.
i=1 stands for i input B with higher amplitude.
Index j represents different trials
The notebook read_signal.ipynb allows to read and plot the spatio-temporal profile
