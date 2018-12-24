#!/bin/bash
set -e

sudo apt-get update -qq;
sudo apt-get install libopenblas-base -y;
sudo apt-get install libopenblas-dev -y;