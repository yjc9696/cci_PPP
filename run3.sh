python codes/data_analyze/split_dataset.py --data_dir mouse_small_intestine4 --train_cluster 4 6 7 8 10 11 12 13 14 16 18 --test_cluster 1 2 3 5 9 15 17 19
python codes/data_analyze/all_pairs.py --data_dir mouse_small_intestine4 --dataset train_dataset --clusters 4 6 7 8 10 11 12 13 14 16 18
python codes/data_analyze/all_pairs.py --data_dir mouse_small_intestine4 --dataset test_dataset --clusters 1 2 3 5 9 15 17 19
