# cci
cell cell interaction
run2.sh add bn
run1 run3 no bn

final version dataset:
train_dataset4:
train_dataset3:

mousev1.pth: precision:0.8  recall:0.7
python ./code/train_mammary.py --dataset 1189 --train_dataset train_dataset2_fake --test_dataset test_dataset2_fake \
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


train_dataset: junk are randomly chosen.
train_dataset2: 选择负样本，基因交叉后，随机选取共同基因最少5个细胞中的一个
train_dataset3: 选择负样本，基因交叉后，随机选取共同基因最少5个细胞中的一个 和2的cluster不一样
train_dataset2_3: 只选了1个基因最少的细胞，所以有很多0


add dev branch