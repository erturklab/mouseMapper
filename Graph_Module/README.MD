# Graph extraction module

## Requirements
Graph extraction requires the installation of [pi2/itl2](https://github.com/arttumiettinen/pi2). After installation, please change the config file to the corresponding pi2 location.

## Usage
Please fill out the paths in the configuration file. Graph extraction can run in multi- and single-processing ways, and can iterate patch-wise over an image. Fusion of graphs is currently still work in progress.
After adjusting the config file, the graph extraction can be run with ```python __main__.py```.
This will output both a .vtk file of the graph, pickledumps containing dictionaries of nodes and edges as well as figures of comparisons between groups.
The expected runtime for the example data (an infraorbital nerve) is ~4 minutes.
