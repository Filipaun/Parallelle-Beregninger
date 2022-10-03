echo
echo "Comparing parallel solution with reference data"
echo "------------------------------------------------------------"
echo "Grid size: 1024, Max iteration: 100000, Snapshot frequency: 1000"
echo
echo "Checking noneven grids"
echo "------------------------------------------------------------------"
echo
for i in 3 5 6 7
do
    echo "Running with $i processes:"
    mpirun -n $i --oversubscribe ./parallel 1>/dev/null
    ./check/compare_solutions 1024 data/00050.bin check/references/n1024/00050.bin
    echo
done
