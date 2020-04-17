python ./codes/train_mammary.py --data_dir mouse_small_intestine \
--cell_data_path mouse_small_intestine_1189_data.csv \
--ligand_receptor_gene mouse_ligand_receptor_pair.csv \
--train_dataset train_dataset --test_dataset test_dataset \
--gpu 0 \
--dense_dim 200 \
--hidden_dim 50 \
--lr 1e-4 \
--n_epochs 10000 \
--batch_size 32 \
--dropout 0.1 \
--loss_weight 1 1 \
--n_layers 2 \
--pretrained_model_path checkpoints/best_modelv3.pth \
--load_pretrained_model 0 \
--save_model_path checkpoints/best_modelv3.pth \
--just_train 0 \
--each_dataset_size 0


