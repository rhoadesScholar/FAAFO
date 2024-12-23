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

# echo "Baseline student training..."
# cd student_baseline
# # cd ../student_baseline
# sh clean.sh
# python student_baseline.py

# cd ../student_noAugment
# sh clean.sh
# python student_noAugment.py

# cd ../student_gtAugment
# sh clean.sh
# python student_gtAugment.py

echo "joint students and teachers..."
cd ../joint_train
# cd joint_train
sh clean.sh
python joint_train.py

# Loop until no jobs are running or pending
while bjobs 2>&1 | grep -q 'RUN\|PEND'; do
    echo "Jobs are still running or pending..."
    sleep 560  # Wait for 5 minutes before checking again
done

echo "initial student and teacher training is done!"

echo "Knowledge committee expansion..."
cd ../ensemble_expanded
# cd ensemble_expanded
sh clean.sh
python ensemble_expanded.py


echo "Knowledge committee expansion with alternating BCE..."
cd ../ensemble_alternating
sh clean.sh
python ensemble_alternating.py

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