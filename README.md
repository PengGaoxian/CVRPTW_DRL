### 生成数据
#### 验证数据集
```bash
python generate_data.py --problem cvrptw --name validation --seed 4321
```
#### 测试数据集
```bash
python generate_data.py --problem cvrptw --name test --seed 1234
```
### 训练
```bash
python run.py --problem cvrptw --graph_size 20 --baseline rollout --run_name cvrptw20_rollout --val_dataset data/cvrptw/cvrptw20_validation_seed4321.pkl
```
### 评估：
#### 贪婪策略
##### 使用自己训练的模型
```bash
python eval.py data/cvrptw/cvrptw20_test_seed1234.pkl --model outputs/cvrptw_20/cvrptw20_rollout/epoch-99.pt --decode_strategy greedy
```
#### 采样策略
##### 使用自己训练的模型
```bash
python eval.py data/cvrptw/cvrptw20_test_seed1234.pkl --model outputs/cvrptw_20/cvrptw20_rollout/epoch-99.pt --decode_strategy sample --width 1280 --eval_batch_size 1
```