

For running mexGPUall.m, do this:

```python
module add openmind/cuda/9.1
module add openmind/cudnn/9.1-7.0.5
module add openmind/gcc/5.3.0
module add mit/matlab/2018b

matlab mexGPUall
```

This should run for a minute then say it completed successfully.