python ./codes/train_mammary.py --dataset 1189 --train_dataset train_dataset2_fake --test_dataset test_dataset2_fake \
--tissue small_intestine --gpu 0 \
--dense_dim 400 \
--hidden_dim 200 \
--lr 1e-6 \
--n_epochs 10000 \
--batch_size 32 \
--dropout 0.1 \
--loss_weight 1 \
--n_layers 1 \
--pretrained_model_path checkpoints/mousev1.pth \
--load_pretrained_model 1 \
--save_model_path checkpoints/mousev1.pth


