# Bundle Adjustment

## Build
CMake 3.14 or higher is required. Download it using e.g. homebrew/linuxbrew

```
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=/home/linuxbrew/.linuxbrew/lib/eigen3 
make -j4
```

Optional: `-DCUDA_V=10.2 -DDOWNLOAD_DATASETS=OFF`