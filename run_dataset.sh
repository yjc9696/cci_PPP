# python codes/data_analyze/all_pairs.py --data_dir small_intestine_bone_marrow --dataset train_dataset
# python codes/data_analyze/all_pairs.py --data_dir small_intestine_bone_marrow --dataset test_dataset

# python codes/data_analyze/split_dataset.py --train_cluster 1 5 6 7 8 10 12 16 27 --test_cluster 2 3 4 9 11 13 42
# python codes/data_analyze/all_pairs.py --data_dir mouse_bone_marrow --dataset train_dataset --clusters 1 5 6 7 8 10 12 16 27
# python codes/data_analyze/all_pairs.py --data_dir mouse_bone_marrow --dataset test_dataset --clusters 2 3 4 9 11 13 42

python codes/data_analyze/split_dataset.py --data_dir mouse_small_intestine4 --test_cluster 4 6 7 8 10 11 12 13 14 16 18 --train_cluster 1 2 3 5 9 15 17 19
python codes/data_analyze/all_pairs.py --data_dir mouse_small_intestine4 --dataset test_dataset --clusters 4 6 7 8 10 11 12 13 14 16 18
python codes/data_analyze/all_pairs.py --data_dir mouse_small_intestine4 --dataset train_dataset --clusters 1 2 3 5 9 15 17 19


# python codes/data_analyze/split_dataset.py --data_dir mouse_bone_marrow2 \
# --cell_data mouse_bone_marrow_911_data.csv \
# --cell_cluster mouse_bone_marrow_911_cellcluster.csv \
# --cluster_cluster_interaction_enriched mouse_bone_marrow_911_cluster_cluster_interaction_enriched.csv \
# --cluster_cluster_interaction_depleted mouse_bone_marrow_911_cluster_cluster_interaction_depleted.csv \
# --ligand_receptor_gene mouse_ligand_receptor_pair.csv \
# --train_cluster 1 3 5 6 9 10 12 14 16 27 \
# --test_cluster 2 4 7 8 11 13 15 42


# python codes/data_analyze/all_pairs.py --data_dir mouse_bone_marrow2 \
# --cell_data mouse_bone_marrow_911_data.csv \
# --cell_cluster mouse_bone_marrow_911_cellcluster.csv \
# --cluster_cluster_interaction_enriched mouse_bone_marrow_911_cluster_cluster_interaction_enriched.csv \
# --cluster_cluster_interaction_depleted mouse_bone_marrow_911_cluster_cluster_interaction_depleted.csv \
# --ligand_receptor_gene mouse_ligand_receptor_pair.csv \
# --dataset train_dataset \
# --clusters 1 3 5 6 9 10 12 14 16 27 \
# --analyze_file mouse_bone_marrow_911_analyze.csv


# python codes/data_analyze/all_pairs.py --data_dir mouse_bone_marrow2 \
# --cell_data mouse_bone_marrow_911_data.csv \
# --cell_cluster mouse_bone_marrow_911_cellcluster.csv \
# --cluster_cluster_interaction_enriched mouse_bone_marrow_911_cluster_cluster_interaction_enriched.csv \
# --cluster_cluster_interaction_depleted mouse_bone_marrow_911_cluster_cluster_interaction_depleted.csv \
# --ligand_receptor_gene mouse_ligand_receptor_pair.csv \
# --dataset test_dataset \
# --clusters 2 4 7 8 11 13 15 42 \
# --analyze_file mouse_bone_marrow_911_analyze.csv