# small_intestine_bone_marrow train, num of pairs
# python ./code/data_analyze/generate_gt1.py --dataset train_dataset --clusters 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# small_intestine_bone_marrow test  num of pairs
# python ./code/data_analyze/generate_gt1.py

python ./code/data_analyze/generate_gt1.py --data_dir mouse_small_intestine2 --dataset train_dataset --pos_or_neg pos --clusters 1 3 5 7 9 10 13 14 15 16 17 18 19 20
python ./code/data_analyze/generate_gt1.py --data_dir mouse_small_intestine2 --dataset test_dataset --pos_or_neg pos --clusters 2 4 6 8 11 12
python ./code/data_analyze/generate_gt1.py --data_dir mouse_small_intestine2 --dataset train_dataset --pos_or_neg neg --clusters 1 3 5 7 9 10 13 14 15 16 17 18 19 20
python ./code/data_analyze/generate_gt1.py --data_dir mouse_small_intestine2 --dataset test_dataset --pos_or_neg neg --clusters 2 4 6 8 11 12