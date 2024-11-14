#!/bin/bash

echo "BCE training..."
cd binary
# cd ../binary
sh clean.sh
python train.py

echo "distance..."
cd ../distance
# cd distance
sh clean.sh
python train.py

echo "signed distance..."
cd ../signed_distance
# cd distance
sh clean.sh
python train.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "Training is done!"

echo "Evaluating accuracy and calibration..."
cd ..
python calibration.py

echo "All done!"