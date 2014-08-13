#!/bin/sh

CC=armv7a-hardfloat-linux-gnueabi-gcc

${CC} -O2 -g -o bench_yuv bench_yuv.c ../simd/jsimd_arm_neon.S -lm || exit 1

echo === Copy and run benchmark on 1GHz Cortex-A7 ===
ssh root@ct killall bench_yuv
scp bench_yuv root@ct:/root/bench_yuv
ssh root@ct /root/bench_yuv 32 1008
ssh root@ct /root/bench_yuv 24 1008
ssh root@ct /root/bench_yuv 16 1008

echo === Copy and run benchmark on 1GHz Cortex-A8 ===
ssh root@mele killall bench_yuv
scp bench_yuv root@mele:/root/bench_yuv
ssh root@mele /root/bench_yuv 32 1008
ssh root@mele /root/bench_yuv 24 1008
ssh root@mele /root/bench_yuv 16 1008

echo === Copy and run benchmark on 1GHz Cortex-A9 ===
ssh root@odroidx killall bench_yuv
scp bench_yuv root@odroidx:/root/bench_yuv
ssh root@odroidx /root/bench_yuv 32 1400
ssh root@odroidx /root/bench_yuv 24 1400
ssh root@odroidx /root/bench_yuv 16 1400

echo === Copy and run benchmark on 1.7GHz Cortex-A15 ===
ssh root@a15 killall bench_yuv
scp bench_yuv root@a15:/root/bench_yuv
ssh root@a15 /root/bench_yuv 32 1700
ssh root@a15 /root/bench_yuv 24 1700
ssh root@a15 /root/bench_yuv 16 1700

echo === Copy and run benchmark on 1.67GHz Krait 300 ===
ssh root@krait killall bench_yuv
scp bench_yuv root@krait:/root/bench_yuv
ssh root@krait /root/bench_yuv 32 1670
ssh root@krait /root/bench_yuv 24 1670
ssh root@krait /root/bench_yuv 16 1670
