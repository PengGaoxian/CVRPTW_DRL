#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


def run(opts):

    # 打印运行的参数
    pp.pprint(vars(opts))

    # 设置随机种子
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

####################  创建输出目录'outputs/cvrp_20/cvrp_20_20191212T161616和保存运行参数的文件'args.json'  ####################
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
#########################################################################################################################
    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

#########################  加载问题  #########################
    problem = load_problem(opts.problem)
#############################################################

####################  加载模型参数  ####################
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path) # 从load_path中加载模型参数到load_data（字典）
######################################################

#####################################  初始化模型  #####################################
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim, # 嵌入向量维度
        opts.hidden_dim, # 隐藏层向量维度
        problem, # 问题实例
        n_encode_layers=opts.n_encode_layers, # 编码器中隐藏层的层数
        mask_inner=True, # 是否在内部层中使用掩码
        mask_logits=True, # 是否在计算logits时使用掩码
        normalization=opts.normalization, # 归一化类型
        tanh_clipping=opts.tanh_clipping, # 使用tanh裁剪参数的值
        checkpoint_encoder=opts.checkpoint_encoder, # 是否检查点编码器
        shrink_size=opts.shrink_size # 缩小大小以节省内存
    ).to(opts.device)
    # 如果使用cuda并且有多个GPU，则使用DataParallel，在多个GPU上并行运行
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 用load_data中的模型参数覆盖model中的模型参数
    model_ = get_inner_model(model) # model_是对model的内部模型的引用，修改model_就是修改model
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
#########################################################################################


###########################################  初始化baseline  ###########################################
    if opts.baseline == 'exponential': # 指数类型基线
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm': # critic类型基线（只支持TSP）
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout': # rollout类型基线
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])
############################################################################################################

########################################  初始化优化器  ########################################
    optimizer = optim.Adam( # 需要优化的模型参数，以及学习率
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device) # 张量需要在指定设备上进行计算，所以需要确保所有张量都在正确的设备上

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
###############################################################################################

#######################################################  开始训练  #######################################################
    val_dataset = problem.make_dataset(  # 加载或生成验证集
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
        distribution=opts.data_distribution)

    if opts.resume: # 如果是恢复训练，则从上次训练的epoch开始
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only: # 如果只是评估模型，则只进行评估
        validate(model, val_dataset, opts)
    else: # 否则进行训练
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model, # 模型
                optimizer, # 优化器
                baseline, # 基线（强化学习）
                lr_scheduler, # 学习率调度器
                epoch, # 当前epoch
                val_dataset, # 验证集
                problem, # 问题实例
                tb_logger, # tensorboard日志
                opts # 运行参数
            )
#########################################################################################################################

if __name__ == "__main__":
    run(get_options())
