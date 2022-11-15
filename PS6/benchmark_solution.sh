N_list=(2 4 8 16 32 64 128 256 512 1024)
echo "Bencmarking bad CUDA GEMM"
for i in "${N_list[@]}"
do
    echo "------------------------------------------------------------"
    echo "Matrix size N = $i"
    echo "------------------------------------------------------------------"
    echo "Serial"
    time $(./serial $i 1>/dev/null)
    echo
    echo "CUDA"
    time $(./parallel_cuda $i 1>/dev/null)
    echo
    echo "cuBLAS"
    time $(./parallel_cublas $i 1>/dev/null)
    echo
done