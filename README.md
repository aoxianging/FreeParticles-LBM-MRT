1. Change the current working directory to the MF-LBM-CUDA
```bash
$ cd MF-LBM-CUDA/
```
2. Edit the Makefile
 Use your preferred directory to modify the following parameters in the Makefile to match your CUDA toolkit installation and your GPU compute capability.
  ```makefile
  COMPUTE_CAPABILITY := 
  CUDA_LIBRARY_LOCATION := 
  CUDA_LNK_PATH := 
  ```
 Also, choose which type of build to be compiled (release/debug).
  ```makefile
   BUILD :=
  ```
  Optionally, add any additional flags to be passed to the GCC or the NVCC compilers
  ```makefile
  CPPFLAGS_ADD := 
  CUFLAGS_ADD := 
  ```
3. Select the solver precision
 Use your preferred editor to modify the following parameter in the header file "solver_precision.h"
```c++
/* Select solver precision: SINGLE_PRECISION / DOUBLE_PRECISION */
#define PRECISION (SINGLE_PRECISION)	
```
4. Compile the program using GNU Make
```bash
$ Make
```
5. Run the program
```bash
$ ./LBM_CUDA
```

该代码的目标是实现
功能：剪切流中球形颗粒受力分析。
1. 所有边界均使用郭照立2002年提出的非平衡态外推实现，其中进出口（y方向）为各处具有相同剪切率的速度边界条件，前后边界（x方向）也是具有速度梯度的速度边界。上下边界（z方向）为固定速度的速度边界条件。
2. Re数使用最大速度计算。
3. 代码中球形粒子在计算几何的中心位置。
4. 代码为D3Q19，SRT，平衡态分布函数为最原始的形式，没有不可压修正。
