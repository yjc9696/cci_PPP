python ./code/train_mammary.py --dataset 1189 --train_dataset train_dataset2 --test_dataset test_dataset2 \
--tissue small_intestine --gpu 0 \
--dense_dim 150 \
--hidden_dim 40 \
--lr 1e-4 \
--n_epochs 500 \
--batch_size 256 \
--dropout 0.2 \
--loss_weight 5 1 \
--n_layers 2 \
--pretrained_model_path checkpoints/best_modelv1.pth \
--load_pretrained_model 1 \
--save_model_path checkpoints/best_modelv1.pth \
--just_train 0 \
--using_mmd 1 \
--each_dataset_size 500


