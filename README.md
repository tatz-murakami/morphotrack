# morphotrack

Reconstruct morphogenic tracks.
Relevant paper: 
> Murakami and Heintz. Multiplexed and scalable cellular phenotyping toward the standardized three-dimensional human neuroanatomy. bioRxiv, 2022


### Software prerequisites 

Tested on Ubuntu 20.04 LST with the following versions of software.
- Cmake 3.21.3
- Python 3.8.12
- CUDA Toolkit 11.6
- [deform](https://github.com/simeks/deform)
- ISPC 1.16.1

### Source data prerequisites 

1. 10 micrometer voxel nuclei and blood vessel (stained with anti-alpha-smooth-muscle actin antibody)
2. binary mask for the tissue, the white matter and the layer 1
3. the cellular coordinates in 10 micrometer unit

For 2, use manual/automated segmentation tool such as [labkit](https://imagej.net/plugins/labkit/).

For 3, CNN-based segmentation tools (e.g. [cellpose](https://github.com/MouseLand/cellpose) or [stardist](https://github.com/stardist/stardist)) show high performance but not limited to them. 


### Installing python packages
```
conda env create -f environment.yml
```


### Usages
See the ipython notebooks.
