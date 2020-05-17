python ./codes/train_mammary_norm.py --data_dir mouse_small_intestine3 \
--cell_data_path mouse_small_intestine_1189_data.csv \
--ligand_receptor_gene mouse_ligand_receptor_pair.csv \
--train_dataset train_dataset \
--test_dataset test_dataset \
--gpu 1 \
--dense_dim 100 \
--hidden_dim 50 \
--aggregator_type mean \
--lr 1e-3 \
--n_epochs 200 \
--batch_size 64 \
--dropout 0.2 \
--loss_weight 1 1 \
--n_layers 2 \
--pretrained_model_path checkpoints/best_modelv4.pth \
--load_pretrained_model 0 \
--save_model_path checkpoints/best_modelv4.pth \
--score_limit 50 \
--score_type score \
--using_ligand_receptor 1 \
--using_func_nodes 1 \
--reduction_ratio 20 \
--evaluate_percentage 0.7
