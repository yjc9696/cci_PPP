# cci
cell cell interaction


add dev branch
dev: 使用num of pairs选择的正负样本
dev2: 进行新的工作迭代

1. 运行split_dataset将mouse small intesine分成测试集和训练集，需要先建好train_dataset test_dataset文件夹，每个文件夹会有cell-data和cell-cluster文件，将ligand-receptor-gene，cluster-interaction文件也拷贝进traindataset和testdataset中。
    test_cluster = [2, 4, 6, 8, 11, 12]
    train_cluster = [1, 3, 5, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]

<!-- 2. 运行generate_gt1.py生成训练所需的正负样本，在projectpath/data/mouse_small_intestine/train_dataset/data 文件夹下

mouse_small_intestine: train, test 都按照num pairs选择基因和负样本
mouse_small_intestine2: 选择cluster-cluster的所有pairs
mouse_small_intestine3: train test 都按照num pairs选择基因和负样本，
mouse_small_intestine4: train按照num pairs选择基因,test使用cluster-cluster所有pairs

small_intestine_bone_marrow: traindataset是small intestine， test dataset是 bone marrow -->
2. 运行all_pairs.py生成analyze文件


## Record
python ./codes/train_mammary.py --data_dir mouse_small_intestine2 \
--cell_data_path mouse_small_intestine_1189_data.csv \
--ligand_receptor_gene mouse_ligand_receptor_pair.csv \
--train_dataset train_dataset \
--test_dataset test_dataset \
--gpu 0 \
--dense_dim 80 \
--hidden_dim 40 \
--lr 1e-4 \
--n_epochs 80 \
--batch_size 32 \
--dropout 0.1 \
--loss_weight 1 1 \
--n_layers 2 \
--pretrained_model_path checkpoints/best_modelv4.pth \
--load_pretrained_model 0 \
--save_model_path checkpoints/best_modelv4.pth \
--score_limit 60

Epoch 0058: precesion 0.87878, recall 0.773017, train loss: 1500.4022216796875
Epoch 0058: precesion 0.87156, recall 0.767954, vali loss: 1529.3721923828125
Epoch 0058: precesion 0.84440, recall 0.064344, test loss: 4264.83544921875
Epoch 0059: precesion 0.87450, recall 0.798288, train loss: 1464.9393310546875
Epoch 0059: precesion 0.87032, recall 0.794293, vali loss: 1496.090087890625
Epoch 0059: precesion 0.84441, recall 0.069838, test loss: 4022.41650390625
Epoch 0060: precesion 0.86940, recall 0.807846, train loss: 1456.2225341796875
Epoch 0060: precesion 0.86405, recall 0.804653, vali loss: 1487.135009765625
Epoch 0060: precesion 0.82558, recall 0.071862, test loss: 3806.470947265625
Epoch 0061: precesion 0.86012, recall 0.831708, train loss: 1445.3499755859375
Epoch 0061: precesion 0.85625, recall 0.826251, vali loss: 1480.0936279296875
Epoch 0061: precesion 0.81613, recall 0.073164, test loss: 4121.72314453125
Epoch 0062: precesion 0.85839, recall 0.821521, train loss: 1463.853271484375
Epoch 0062: precesion 0.85307, recall 0.817120, vali loss: 1495.91748046875
Epoch 0062: precesion 0.82071, recall 0.076778, test loss: 3594.89697265625