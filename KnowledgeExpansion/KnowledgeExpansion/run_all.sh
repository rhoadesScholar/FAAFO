#!/bin/bash

# echo "Pretraining teachers..."
# cd pretrain_teacher
# sh clean.sh
# python pretrain_teacher.py

# # Loop until no jobs are running or pending
# while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
#     echo "Jobs are still running or pending..."
#     sleep 560  # Wait for 5 minutes before checking again
# done

# echo "Pretraining teachers is done!"


echo "split students and teachers..."
cd split_train
sh clean.sh
python split_train.py

cd ../student_baseline
sh clean.sh
python student_baseline.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "split training is done!"


echo "Knowledge committee expansion..."
cd ../ensemble_expanded
sh clean.sh
python ensemble_expanded.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "Knowledge ensemble_expanded is done!"


echo "Evaluating accuracy and calibration..."
cd ..
python calibration.py

echo "All done!"