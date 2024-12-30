
# input is the date Sep_20_g0

mkdir "$1"

rsync -ruv --max-size=1024m c_tang@openmind-dtn.mit.edu:/om2/user/c_tang/Sorting/Data/Faure/NP/"$1"/"$1"_imec1/peaks ./"$1"
#scp -r c_tang@openmind-dtn.mit.edu:/om2/user/c_tang/Sorting/Data/Faure/NP/"$1"/"$1"_imec0/"$1"_t0.imec0.ap.bin ./"$1"/ks_3_output_pre_v6/


