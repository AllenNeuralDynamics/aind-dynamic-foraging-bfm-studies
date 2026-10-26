#!/bin/bash
#SBATCH -p aind
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -J s13_cpu_probe
#SBATCH -o /home/han.hou/logfile/s13_cpu_probe_%j.log
cd /home/han.hou/code/wt/hpc13/probe
JAX_PLATFORMS=cpu /allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu/bin/python -W ignore cpu_probe.py
