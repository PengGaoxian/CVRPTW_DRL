import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from problems.vrp.problem_vrp import CVRPTWDataset


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost, violated_time, route_length = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    avg_violated_time = violated_time.mean()
    avg_route_length = route_length.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    print('Validation overall avg_violated_time: {} +- {}'.format(
        avg_violated_time, torch.std(violated_time) / math.sqrt(len(violated_time))))
    print('Validation overall avg_route_length: {} +- {}'.format(
        avg_route_length, torch.std(route_length) / math.sqrt(len(route_length))))

    return avg_cost, avg_violated_time, avg_route_length


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, ll, violated_time, violated_node, route_length = model(move_to(bat, opts.device))
        # return cost
        return torch.stack((cost.data.cpu(), violated_time.data.cpu(), route_length.data.cpu()), dim=1)

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0).unbind(dim=1)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    # 包装数据集：接收数据集作为参数，返回数据集和基线值（eval后的值）
    random_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    """
    仅用于导出节点数量为20的训练数据集到data目录下
    """
    save_random_dataset = 0
    if save_random_dataset:
        from utils.data_utils import save_dataset
        modified_dataset = []
        for dataset_index in range(512*100):
            # 拼接list变量，用于保存数据集

            dataset= (
                random_dataset[dataset_index]['depot'].numpy().tolist(), # 将torch张量转换为numpy数组，然后转换为列表
                random_dataset[dataset_index]['loc'].numpy().tolist(),
                (random_dataset[dataset_index]['demand'].numpy()*30).tolist(),
                30,
                (random_dataset[dataset_index]['tw'].numpy()*8).tolist(),
                8
            )
            modified_dataset.append(dataset)
        save_dataset(modified_dataset, 'data/cvrptw_20_train.pkl')

    training_dataset = baseline.wrap_dataset(random_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward, avg_violated_time, avg_route_length = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
        tb_logger.log_value('val_avg_violated_time', avg_violated_time, step)
        tb_logger.log_value('val_avg_route_length', avg_route_length, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch, # loc、demand、depot
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch) # 解包数据集：接收批次数据作为参数，返回数据和基线值（None）
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    # cost：经过模型model采样（sample）处理后的数据序列的路径总长度(batch_size, )
    # log_likelihood：经过模型model处理后的数据序列的总似然对数(batch_size, )
    cost, log_likelihood, violated_time_num, violated_node_num, route_length = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    """
    评估方法：接收批量数据(字典,包含loc、depot、demand)和总成本作为参数
    首轮Warmup时，batch中只有dataset数据，返回批量总成本的均值作为基线值和总成本(0)（因为没有损失）
    第二轮开始，batch中包含对dataset数据和dataset数据的greedy策略的评估值，返回评估值作为基线值和总成本(0)（因为没有损失）
    """
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    """
    cost是采样（sample）策略得到的目标值，bl_val是贪婪（greedy）策略得到的目标值，
    因为训练过程中采样策略的结果会比贪婪策略的结果差，所以reinforce_loss是负值，
    但是随着训练次数越来越多，reinforce_loss会逐渐收敛到0
    """
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, violated_time_num, violated_node_num, route_length)
