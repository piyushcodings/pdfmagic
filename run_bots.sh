#!/bin/bash
# Start both bots in the background
python main.py &
python main2.py &

# Wait for both to finish
wait
