mkdir build
cd build
cmake ..
make -j
ctest

if [[ $(SYSTEM_NAME) =~ "YOKOTA_LAB"  ]]; then
    exit 0
fi
