#!/bin/bash

echo "plots PSK dataset 10000 instances uniform between 1 and 4"

# ./main.py -n 1                             -k best            -o results/psk-400-10000-best-ol-final-test            data/psk_400_10000_discrete.txt
./main_cumulative.py -n 30                             -k best            -o results/psk-8-2000-best-ol-cumulative-b2-discrete-test           -d data/psk_8_10000_discrete.txt
# ./main_cumulative.py -n 1                             -k best            -o results/psk-400-10000-best-ol-violin-b40-discrete           -d data/psk_400_10000_discrete.txt
# ./main_adversarial.py -n 3                             -k best            -o results/psk-8-2000-best-ol-adversarial-b2-discrete-test           -d data/psk_8_10000_discrete.txt
