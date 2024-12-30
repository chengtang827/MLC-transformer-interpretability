
# input is the date Sep_20_g0

mkdir "$1"

rsync -ruv --max-size=1024m c_tang@openmind-dtn.mit.edu:/om2/user/c_tang/Sorting/Data/Faure/NP/"$1" ./"$1"
#''

