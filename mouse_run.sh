python ./code/train_mammary.py --dataset 1189 --train_dataset train_dataset2 --test_dataset test_dataset2 \
--tissue small_intestine --gpu 0 \
--dense_dim 400 \
--hidden_dim 200 \
--lr 1e-4 \
--n_epochs 10000 \
--batch_size 320 \
--dropout 0.2 \
--loss_weight 1 \
--n_layers 1 \
--pretrained_model_path checkpoints/best_modelv1.pth \
--load_pretrained_model 0 \
--save_model_path checkpoints/best_modelv1.pth

