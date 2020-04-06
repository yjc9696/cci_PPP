#python ./code/data_analyze/generate_gt_negative.py --dataset train_dataset --junk junk --cluster_cluster_interaction_depleted mouse_small_intestine_1189_cluster_cluster_interaction_depleted.csv
#python ./code/data_analyze/generate_gt_negative.py --dataset train_dataset --junk gt --cluster_cluster_interaction_depleted mouse_small_intestine_1189_cluster_cluster_interaction_enriched.csv
#python ./code/data_analyze/generate_gt_negative.py --dataset test_dataset --junk junk --cluster_cluster_interaction_depleted mouse_small_intestine_1189_cluster_cluster_interaction_depleted.csv
#python ./code/data_analyze/generate_gt_negative.py --dataset test_dataset --junk gt --cluster_cluster_interaction_depleted mouse_small_intestine_1189_cluster_cluster_interaction_enriched.csv

# mouse_small_intestine train, num of pairs
python ./code/data_analyze/generate_gt1.py --dataset train_dataset --clusters 1 3 5 7 9 10 13 14 15 16 17 18 19 20
# mouse_small_intestine test, num of pairs
python ./code/data_analyze/generate_gt1.py --dataset test_dataset --clusters 2 4 6 8 11 12