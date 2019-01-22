# BM3D_GPU
The BM3D image denoising algorithm implemented in CUDA.

The dependencies are:

1) CUDA 7.5 (or newer)

2) Boost library

3) FreeImagePlus library (to read raw format input images)

4) OpenMP

5) chrono library for timing 



after compiling, running command is:

./binary "image file name"  "format" "input directory" "out directory"



The input images can be png files or raw input images from various cameras such as .cr2 , .arw, .nef , .rw2, .sr2 (any input format that FreeImagePlus can read).
The implementation uses tiling for high resolution images to fit the GPU memory and shared memory. 


If you find this implementation useful for your work or experiments, please cite our paper:
https://dl.acm.org/citation.cfm?id=3123941

"Mostafa Mahmoud, Bojian Zheng, Alberto Delmás Lascorz, Felix Heide, Jonathan Assouline, Paul Boucher, Emmanuel Onzon, and Andreas Moshovos. 2017. IDEAL: image denoising accelerator. In Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-50 '17). ACM, New York, NY, USA, 82-95. DOI: https://doi.org/10.1145/3123939.3123941"
