

# Iterate over the arguments using a for loop
for arg in "$@"; do
#    echo "Argument: $arg"
    mkdir "$arg"

    rsync -ruv --max-size=1024m c_tang@openmind-dtn.mit.edu:/om4/group/jazlab/Cheng/sorted/Faure/NP/"$arg"/"$arg"_imec0/kilosort3_corrected_output ./"$arg"
    rsync -ruv --max-size=1024m c_tang@openmind-dtn.mit.edu:/om4/group/jazlab/Cheng/sorted/Faure/NP/"$arg"/"$arg"_imec0/ks3_output_direct ./"$arg"

    rsync -ruv --max-size=10m c_tang@openmind-dtn.mit.edu:/om4/group/jazlab/Cheng/sorted/Faure/NP/"$arg"/"$arg"_imec0/peaks ./"$arg"
    rsync -ruv c_tang@openmind-dtn.mit.edu:/om4/group/jazlab/Cheng/sorted/Faure/NP/"$arg"/"$arg"_imec0/"$arg"_t0.imec0.ap.meta ./"$arg"
    rsync -ruv c_tang@openmind-dtn.mit.edu:/om4/group/jazlab/Cheng/sorted/Faure/NP/"$arg"/"$arg"_t0.nidq.bin ./"$arg"
    rsync -ruv c_tang@openmind-dtn.mit.edu:/om4/group/jazlab/Cheng/sorted/Faure/NP/"$arg"/"$arg"_t0.nidq.meta ./"$arg"

done

# input is the date 20230917_F_g0 20230921_F_g0