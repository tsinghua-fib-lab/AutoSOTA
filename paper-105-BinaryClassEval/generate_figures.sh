python plot.py --demo --subgroup-field "gender" --subgroups "Male" "Female" --minaccuracy 0.75 --style-cycle-offset 8 --average --full-width-average
python plot.py --demo --ece --auc --calibration --maxlogodds 0.1 --nomain --minaccuracy 0.8
python plot.py --demo --calibration --maxlogodds 0.1
python plot.py --demo --average --maxlogodds 0.1