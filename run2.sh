# python codes/data_analyze/all_pairs.py --data_dir small_intestine_bone_marrow --dataset train_dataset
# python codes/data_analyze/all_pairs.py --data_dir small_intestine_bone_marrow --dataset test_dataset

# python codes/data_analyze/split_dataset.py --train_cluster 1 5 6 7 8 10 12 16 27 --test_cluster 2 3 4 9 11 13 42
# python codes/data_analyze/all_pairs.py --data_dir mouse_bone_marrow --dataset train_dataset --clusters 1 5 6 7 8 10 12 16 27
# python codes/data_analyze/all_pairs.py --data_dir mouse_bone_marrow --dataset test_dataset --clusters 2 3 4 9 11 13 42

python codes/data_analyze/split_dataset.py --train_cluster 1 5 6 7 8 10 12 16 27 --test_cluster 2 3 4 9 11 13 42
python codes/data_analyze/all_pairs.py --data_dir mouse_small_intestine3 --dataset train_dataset --clusters 1 5 6 7 8 10 12 16 27
python codes/data_analyze/all_pairs.py --data_dir mouse_small_intestine3 --dataset test_dataset --clusters 2 3 4 9 11 13 42