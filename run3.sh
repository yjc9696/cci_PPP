python ./code/train_mammary.py --dataset 1189 --train_dataset train_dataset3 --test_dataset test_dataset3 \
--tissue small_intestine --gpu 0 \
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
--using_mmd 0 \
--each_dataset_size 10000


