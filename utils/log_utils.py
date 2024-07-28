def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, violated_time_num=None, violated_node_num=None, route_length=None):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    if violated_time_num is not None:
        avg_violated_time_num = violated_time_num.mean().item()
        avg_violated_node_num = violated_node_num.mean().item()
        avg_route_length = route_length.mean().item()
        # Log values to screen
        print('epoch: {}, train_batch_id: {}, avg_cost: {}, avg_violated_time_num: {}, avg_violated_node_num: {}, avg_route_length: {}'.format(epoch, batch_id, avg_cost, avg_violated_time_num, avg_violated_node_num, avg_route_length))
    else:
        print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if violated_time_num is not None:
            tb_logger.log_value('violated_time_num', violated_time_num.mean().item(), step)
            tb_logger.log_value('violated_node_num', violated_node_num.mean().item(), step)
            tb_logger.log_value('route_length', route_length.mean().item(), step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
