#!/bin/bash
g++ -std=c++17 -mavx512f -o /tmp/check_avx512 "$1" && /tmp/check_avx512
