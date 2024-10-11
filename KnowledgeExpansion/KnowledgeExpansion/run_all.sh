#!/bin/bash

echo "Pretraining teachers..."
cd pretrain_teacher
sh clean.sh
python pretrain_teacher.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "Pretraining teachers is done!"

echo "Joint students and teachers..."
cd ../joint_train
sh clean.sh
python joint_train.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "Joint training is done!"

echo "Knowledge expansion..."
cd ../expansion
sh clean.sh
python expansion.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "Knowledge expansion is done!"

echo "Evaluating accuracy and calibration..."
cd ..
python calibration.py

echo "All done!"