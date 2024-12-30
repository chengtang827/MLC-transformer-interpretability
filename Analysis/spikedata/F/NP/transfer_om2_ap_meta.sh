
# input is the date Sep_20_g0

mkdir -p "$1"
scp -r c_tang@openmind-dtn.mit.edu:/om2/user/c_tang/Sorting/Data/Faure/NP/"$1" ./"$1"

#scp -r c_tang@openmind-dtn.mit.edu:/om2/user/c_tang/Sorting/Data/Faure/NP/"$1"/../"$1"_t0.nidq.bin ./"$1"
#scp -r c_tang@openmind-dtn.mit.edu:/om2/user/c_tang/Sorting/Data/Faure/NP/"$1"/"$1"_t0.nidq.meta ./"$1"

#'20230508_F_g0/20230508_F_g0_imec1/20230508_F_g0_t0.imec1.ap.meta'


