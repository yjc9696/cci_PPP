python ./code/train_mammary.py --dataset 1189 --train_dataset train_dataset4 --test_dataset test_dataset4 \
--tissue small_intestine --gpu 1 \
--dense_dim 400 \
--hidden_dim 200 \
--lr 1e-3 \
--n_epochs 1000 \
--batch_size 256 \
--dropout 0.1 \
--loss_weight 6 1 \
--n_layers 2 \
--pretrained_model_path checkpoints/best_modelv1.pth \
--load_pretrained_model 0 \
--save_model_path checkpoints/best_modelv1.pth \
--just_train 0 \
--using_mmd 1 \
--each_dataset_size 100


