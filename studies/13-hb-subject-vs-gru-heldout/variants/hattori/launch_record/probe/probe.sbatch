#!/bin/bash
#SBATCH -p aind
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -J s13_geom_probe
#SBATCH -o /home/han.hou/logfile/s13_cpu_probe_%j.log
cd /home/han.hou/code/wt/hpc13/probe
/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-gpu/bin/python -W ignore geometry_probe.py
