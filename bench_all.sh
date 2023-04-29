#!/bin/bash
mkdir -p ./temp_files/
mkdir -p ./bench_results/
mkdir -p ./bench_results/reading
mkdir -p ./bench_results/writing

echo "READING"
sh ./bench_read_one_channel_i16.sh
sh ./bench_read_one_channel_i16_as_f32.sh
sh ./bench_one_channel_f32.sh
sh ./bench_read_one_channel_f32_as_i16.sh


echo "WRITING"
sh ./bench_write_one_channel_i16.sh
sh ./bench_write_one_channel_i16_as_f32.sh
sh ./bench_write_one_channel_f32.sh
sh ./bench_write_one_channel_f32_as_i16.sh