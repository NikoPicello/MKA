tar -xvf eigen-eigen-323c052e1731.tar.bz2
tar -xvf pybind11-2.5.0.tar.gz
rm -rf build
mkdir build
cd build
cmake ..
make
