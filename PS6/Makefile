.PHONY: clean purge setup check generate

TARGETS:=serial parallel_cuda parallel_cublas matrix-gen
SERIAL_CC:=gcc
PARALLEL_CC:=nvcc

CFLAGS:=-Wall -Werror -Wextra -Wpedantic -Wconversion -O3

# Link the math library
LDFLAGS:=-lm

# Link the cublas library (NVCC only)
PARALLEL_LDFLAGS:=-lcublas


serial: src/matmul_serial.c
	$(SERIAL_CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

parallel_cublas: src/matmul_cublas.cu
	$(PARALLEL_CC) $^ $(LDFLAGS) -lcublas -o $@

parallel_cuda: src/matmul_cuda.cu
	$(PARALLEL_CC) $^ $(LDFLAGS) -lcublas -o $@

matrix-gen: src/matrix_gen.c
	$(SERIAL_CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

setup:
	mkdir data
	cd compare && make

clean:
	- rm $(TARGETS)
	- rm compare/compare

purge: clean
	- rm -rf data

check_cuda: serial parallel_cuda
	./compare.sh cuda

check_cublas: serial parallel_cublas
	./compare.sh cublas

generate: matrix-gen
	./generate-data.sh
