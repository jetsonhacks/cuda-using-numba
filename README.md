# cuda-using-numba
Support code for the JetsonHacks video: CUDA Programming in Python (https://youtu.be/C_WrbBmiTf4).

Numba is a Python module which makes Python code run faster. Numba provides a Just In Time (JIT) Compiler which takes Python byte codes and compiles them to machine code with the help of the LLVM compiler. If you specify that you want CUDA code, then LLVM calls NVVM to create PTX GPU code.

This is demonstration code, not intended for production. None of the code is optimized for its intended purpose. Instead it shows a typical first pass to code speed up.
