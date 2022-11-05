echo
echo "Comparing serial solution with reference data"
echo "-----------------------------------------------------------"
echo "Grid size: 16, Max iteration: 5000, Snapshot frequency: 40"
echo
./serial -n16 1>/dev/null
./check/compare_solutions 16 data/00001.bin check/references/n16/00001.bin
echo "-----------------------------------------------------------"
echo "Grid size: 256, Max iteration: 5000, Snapshot frequency: 40"
echo
./serial -n256 1>/dev/null
./check/compare_solutions 256 data/00001.bin check/references/n256/00001.bin
echo
