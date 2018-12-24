#!/bin/bash
set -e

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  brew update -qq;
  brew install homebrew/science/openblas -y;
fi

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  sudo apt-get update -qq;
  sudo apt-get install libopenblas-base -y;
  sudo apt-get install libopenblas-dev -y;
fi