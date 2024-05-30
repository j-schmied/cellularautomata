#!/usr/bin/env bash

if [ "$#" -gt 3 ]; then
    echo "Usage: $0 <num iterations> [--plot] [--box]"
    exit 1
fi

num_runs=$1
multiverse_id=$(date +"%Y%m%d%H%M%S")

experimental_data_dir=/Volumes/EX4000/DataMedAssist/Experiments/Cellular_Automata/Puliafito_2012_01_17
output_dir=/Volumes/EX4000/DataMedAssist/Eval/cellularautomata_results

if ! [[ "$num_runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid input: Parameter 1 must be a positive integer!"
    exit 1
fi

for ((i=0; i<$num_runs; i++)); do
    # Find valid options in README
    # do NOT alter --multiverse or --well!
    if [[ "$*" == *"--box"* ]]; then
      python -m src.cellularautomaton --multiverse $multiverse_id --well $i --box --exportcsv --output $output_dir --expdata $experimental_data_dir &
    else
      python -m src.cellularautomaton --multiverse $multiverse_id --well $i --colonial --exportcsv --output $output_dir --expdata $experimental_data_dir &
    fi

    sleep 2
done

wait

if [[ "$*" == *"--plot"* ]]; then
  # Adapt experimental_data_dir and output_dir to your needs!
  # NOTE: --exportcsv is necessary for plotting to work!
  if [[ "$*" == *"--box"* ]]; then
    python -m src.utils.plot --path $output_dir/box/multiverse_$multiverse_id --expdata $experimental_data_dir --output pdf --box
  else
    python -m src.utils.plot --path $output_dir/colonial/multiverse_$multiverse_id --expdata $experimental_data_dir --output pdf
  fi
fi

exit 0
