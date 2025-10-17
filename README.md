# Example CUDA code for computing the optimal RMSD and rotation matrix

## Building

```
mkdir build
cd build/
cmake ../
make -j4
```

## Running

Test with the CUDA kernel adapted from Numerical Recipes

`rmsd_cuda_nr ../example_input.txt`

Test with the CUDA kernel adapted from Numerical Recipes (using CUDA graphs)

`rmsd_cuda_nr_graph ../example_input.txt`

Test with CUDA cuSOLVER

`rmsd_cuda_cusolver ../example_input.txt`
