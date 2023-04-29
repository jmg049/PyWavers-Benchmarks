#!/bin/bash

# This script is used to benchmark the performance of the PyWavers (this package), Soundfile, SciPy.IO and TorchAudio(libsox)

# Bench Conditions:
# 1. 1 channel
# 2. 16_bit PCM integer encoding
# 3. Varying sample rate, dataset dependent
# 4. Varying durations, dataset dependent

# Usage:
#1. Put this script in the root directory of the THIS package
#2. Run this script
#3. Profits (in the form of statistics!)

## Quickstart _ GENSPEECH
## About the dataset
## A small verification set of 20 samples taken from the GenSpeech generative speech dataset. The full dataset, alongside this subset are available for download
## from https://github.com/QxLabIreland/datasets
mkdir -p ./bench_results/reading
$(which python) -m pip install -r ./requirements.txt

printf "Starting benchmark _ GenSpeech I16_F32\n"
printf "________________________________________\n"
printf "Benchmarking PyWavers on GenSpeech dataset...\n"
$(which python) benches/reading/bench_read_i16_as_f32.py -o ./bench_results/reading/bench_read_i16_as_f32_timings.json

printf "________________________________________\n"
printf "Benchmarking using track_memory\n"
$(which python) benches/reading/bench_read_i16_as_f32.py --track-memory -o ./bench_results/reading/bench_read_i16_as_f32_track_memory.json

printf "________________________________________\n"
printf "Benchmarking using tracemalloc\n"
$(which python) benches/reading/bench_read_i16_as_f32.py --tracemalloc -o ./bench_results/reading/bench_read_i16_as_f32_tracemalloc.json

printf "________________________________________\n"
printf "Finished benchmark _ GenSpeech\n"
