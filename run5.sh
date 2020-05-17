python ./codes/train_mammary.py --data_dir mouse_small_intestine4 \
--cell_data_path mouse_small_intestine_1189_data.csv \
--ligand_receptor_gene mouse_ligand_receptor_pair.csv \
--train_dataset train_dataset \
--test_dataset test_dataset \
--gpu 0 \
--dense_dim 80 \
--hidden_dim 40 \
--aggregator_type mean \
--lr 1e-4 \
--n_epochs 200 \
--batch_size 32 \
--dropout 0.2 \
--loss_weight 1 1 \
--n_layers 2 \
--pretrained_model_path checkpoints/best_modelv5.pth \
--load_pretrained_model 0 \
--save_model_path checkpoints/best_modelv5.pth \
--score_limit 60 \
--score_type mask_num \
--using_ligand_receptor True \
--using_func_nodes True \
--reduction_ratio 20 \
--evaluate_percentage 0.63
