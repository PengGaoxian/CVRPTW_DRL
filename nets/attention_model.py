import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor # 经过编码器处理后的节点的嵌入向量(batch_size, graph_size+1, embedding_dim)
    context_node_projected: torch.Tensor # 先对经过编码器处理后的节点嵌入的坐标均值的映射(batch_size, 1, embedding_dim)
    glimpse_key: torch.Tensor # (n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
    glimpse_val: torch.Tensor # (n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
    logit_key: torch.Tensor # (batch_size, 1, graph_size=21, embedding_dim)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

# 输入：input（输入），return_pi（是否返回输出序列）；输出：cost（成本），ll（对数似然）
class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim, # 嵌入向量维度
                 hidden_dim, # 隐藏层向量维度
                 problem, # 问题实例
                 n_encode_layers=2, # 编码器中隐藏层的层数
                 tanh_clipping=10., # 使用tanh裁剪参数的值
                 mask_inner=True, # 是否在内部层中使用掩码
                 mask_logits=True, # 是否在计算logits时使用掩码
                 normalization='batch', # 归一化类型
                 n_heads=8, # 多头注意力机制中的头数
                 checkpoint_encoder=False, # 是否检查点编码器
                 shrink_size=None): # 缩小大小以节省内存
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None # 解码类型
        self.temp = 1.0 # softmax温度
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = (problem.NAME == 'cvrp' or problem.NAME == 'sdvrp')
        self.is_cvrptw = (problem.NAME == 'cvrptw')
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads # 多头注意力机制中的头数
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

###################################  问题的上下文参数  ###################################
        if self.is_vrp or self.is_cvrptw or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 3 # 上一个节点的嵌入向量 + 剩余容量 / 剩余长度 / 剩余奖励 + 到达节点的时间 + 车辆速度

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            if self.is_cvrptw:
                node_dim = 5 # x, y, demand, tw_open, tw_close
            else: # cvrp or sdvrp
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim) # 仓库节点的特殊嵌入投影
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
#########################################################################################

        self.init_embed = nn.Linear(node_dim, embedding_dim) # 初始化嵌入
        # 生成图注意力编码器对象，包括节点的嵌入向量和多头注意力层
        self.embedder = GraphAttentionEncoder( # 图注意力编码器
            n_heads=n_heads, # 多头注意力机制中的头数
            embed_dim=embedding_dim, # 嵌入向量维度
            n_layers=self.n_encode_layers, # 编码器中隐藏层的层数
            normalization=normalization # 归一化类型
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input)) # 获取经过注意力机制处理后的节点的嵌入向量，嵌入向量的形状为(batch_size, graph_size+1, embedding_dim)

        # input：字典，包含depot、loc、demand
        # embeddings：(batch_size, graph_size+1, embedding_dim) 经过编码器处理后的节点的嵌入向量
        # _log_p：(batch_size, steps, graph_size+1) 对数概率
        # pi：(batch_size, steps) 输出序列
        _log_p, pi = self._inner(input, embeddings) # 执行解码步骤，获取对数概率和输出序列

        # cost：路径pi的总长度(batch_size, )
        # mask：None

        cost, mask, violated_time_num, violated_node_num, route_length = self.problem.get_costs(input, pi) # 获取一个批量数据的总代价(batch_size,)和掩码
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        # 路径pi的总对数似然(batch_size, )
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi, violated_time_num

        return cost, ll, violated_time_num, violated_node_num, route_length

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        if self.is_vrp or self.is_cvrptw or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_cvrptw:
                features = ('demand', 'tw')
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')

            """
            这里是为了将features中的2维特征转换为3维特征，以便与loc拼接
            """
            # 深拷贝input为input_3dim
            input_3dim = input.copy()
            for feat in features:
                if input[feat].dim() == 2:
                    input_3dim[feat] = input[feat][:, :, None]

            return torch.cat( # depot、loc、demand信息的嵌入向量的拼接(batch_size, graph_size=21, embedding_dim)
                                (
                                            self.init_embed_depot(
                                                input['depot'] # 将(batch_size, 2)--init_embed_depot--> (batch_size, embedding_dim)
                                            )[:, None, :], # (batch_size, embedding_dim)--扩展维度-->(batch_size, 1, embedding_dim)
                                            self.init_embed(
                                                torch.cat(
                                                    (
                                                        input['loc'], # (batch_size, graph_size=20, 2)
                                                        *(input_3dim[feat] for feat in features) # (batch_size, demand_dim=20, 1)
                                                    ),
                                                    -1 # 将local和demand沿最后一个维度拼接成(batch_size, graph_size=20, 3)
                                                )
                                            ) # 将loc和demand拼接后的张量(batch_size, graph_size=20, 3) 转换为(batch_size, graph_size=20, embedding_dim)
                                        ),
                                1 # 将(batch_size, 1, embedding_dim)和(batch_size, graph_size=20, embedding_dim)沿第二个维度拼接成(batch_size, graph_size=21, embedding_dim)
                            )
        # TSP
        return self.init_embed(input)

    # 输入input：(batch_size, graph_size, node_dim) 输入节点特征，包括depot、loc、demand
    # 输入embeddings：(batch_size, graph_size+1, embedding_dim) 经过编码器处理后的节点的注意力嵌入向量
    # 输出outputs：(batch_size, steps, graph_size+1) 对数概率
    # 输出sequences：(batch_size, steps) 输出序列
    def _inner(self, input, embeddings): # embedding为经过解码器后的节点嵌入向量

        outputs = []
        sequences = []

        state = self.problem.make_state(input) # 根据input（depot、loc、demand）生成问题状态

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings) # 计算embeddings的固定上下文，得到AttentionModelFixed对象，其中包含embeddings, fixed_context, multihead_glimpse_key, multihead_glimpse_val, logit_key

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state) # 计算state中当前节点的下一组节点的对数概率和掩码(batch_size, 1, graph_size=21)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            # 根据对数概率和掩码选择下一个节点的索引，如果是greedy解码，则直接选择概率最大的节点，如果是sampling解码，则根据概率采样一个节点
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension (batch_size, )

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            # 0.9的概率选择概率最大的节点，0.1的概率随机选择一个节点
            # if torch.rand(1)[0] < 0.9:
            #     _, selected = probs.max(1)
            # else:
            #     selected = probs.multinomial(1).squeeze(1)

            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        """
        计算固定的上下文参数，包括节点的嵌入向量、固定上下文参数等信息
        :param embeddings: (batch_size, n_loc+1, embedding_dim) 经过编码器处理后的节点嵌入向量
        :param num_steps: 解码步数（解码器一次性输出多少个节点）
        :return: AttentionModelFixed对象（embeddings, fixed_context, multihead_glimpse_key, multihead_glimpse_value, logit_key）
        """

        # 为了提高效率，embeddings的固定上下文投影仅计算一次
        graph_embed = embeddings.mean(1) # (batch_size, embedding_dim) 所有节点的嵌入向量的均值
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :] # (batch_size, 1, embed_dim) 先对所有坐标的均值做映射，然后扩展维度

        # 节点嵌入向量映射成多头注意力机制的键、值和logits
        # 先将embeddings(batch_size, 21, embedding_dim)扩展成(batch_size, 1, 21, embedding_dim)，
        # 然后将其映射成(batch_size, 1, 21, 3 * embedding_dim),
        # 最后将其分成3个张量，形状为(batch_size, 1, 21, embedding_dim)
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),  # (n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
            self._make_heads(glimpse_val_fixed, num_steps),  # (n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
            logit_key_fixed.contiguous() # (batch_size, 1, graph_size=21, embedding_dim)
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        """
        :param fixed: AttentionModelFixed，其中包含embeddings, fixed_context, multihead_glimpse_key, multihead_glimpse_val, logit_key
        :param state: 问题状态，包括当前节点、已访问节点、已使用容量等信息
        :param normalize: 是否进行softmax归一化
        :return: log_p: (batch_size, 1, graph_size) 当前节点状态state计算下一组可能节点的log概率（归一化的query与logits_K的相似度）
        :return: mask: (batch_size, 1, graph_size) 掩码
        """
        # Compute query = context node embedding
        # (batch_size, 1, embedding_dim)所有节点的均值的嵌入向量 + (batch_size, num_steps, embedding_dim)当前节点的嵌入向量和剩余容量的拼接
        query = fixed.context_node_projected + \
                self.project_step_context(
                    self._get_parallel_step_context(fixed.node_embeddings, state) # 当前节点的嵌入向量和剩余容量的拼接映射成(batch_size, num_steps, embedding_dim+2)
                ) # 将(batch_size, num_steps, embedding_dim+1)映射成(batch_size, num_steps, embedding_dim)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state) # 返回fixed中的multihead_glimpse_key、multihead_glimpse_val和logit_key

        # 计算当前节点下所有节点（depot节点和loc节点）的掩码
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        # 获取query与logit_K的相似度，以及query的注意力值向量
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        # log_p：通过softmax归一化的query与logit_K的相似度
        # mask：掩码
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        功能：从state中获取当前节点，然后从embeddings中获取当前节点的嵌入向量，最后将当前节点的嵌入向量和剩余容量的拼接
        
        :param embeddings: (batch_size, graph_size, embed_dim) 编码器输出的节点嵌入向量
        :param state: 问题状态，包括当前节点、已访问节点、已使用容量等信息
        :param from_depot: 是否从仓库开始
        :return: (batch_size, num_steps, embedding_dim+1) 当前节点的嵌入向量、剩余容量、到达时间、车辆速度的拼接
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_vrp or self.is_cvrptw:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat( # 当前节点的嵌入向量和剩余容量的拼接(batch_size, num_steps, embedding_dim+1)
                    (
                        torch.gather( # 通过当前节点的索引获取节点的嵌入向量
                            embeddings, # (batch_size, graph_size=21, embedding_dim=128)
                            1, #
                            current_node.contiguous() # (1024, 1)
                                .view(batch_size, num_steps, 1) # (1024, 1, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1)) # (1024, 1, 128) 将当前节点的索引扩展成与嵌入向量相同的维度
                        ).view(batch_size, num_steps, embeddings.size(-1)), # (1024, 1, 128) 当前节点的嵌入向量
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None], # (1024, 1, 1) 剩余容量
                        state.cur_arrival_time[:, :, None], # (1024, 1, 1) 当前节点的到达时间
                        state.vehicle_speed[:, :, None] # (1024, 1, 1) 车辆的速度
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        功能：计算query的多头注意力值向量和注意力值向量与logit_K的相似度
        :param query: 查询向量，即节点的平均嵌入向量+当前节点的位置和剩余容量拼接的嵌入向量，形状为(batch_size, num_steps=1, embedding_dim)
        :param glimpse_K: 所有节点经过编码器处理后的嵌入向量的key的多头注意力嵌入向量，形状为(n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
        :param glimpse_V: 所有节点经过编码器处理后的嵌入向量的value的多头注意力嵌入向量，形状为(n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
        :param logit_K: 所有节点经过编码器处理后的注意力嵌入向量，形状为(batch_size, 1, graph_size=21, embedding_dim)
        :param mask: 掩码，形状为(batch_size, num_steps=1, graph_size=21)
        :return logits: 注意力值向量与logit_K的相似度，形状为(batch_size, num_steps=1, graph_size=21)
        :return glimpse: 注意力值向量，形状为(batch_size, num_steps, 1, embedding_dim)
        """

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        # 将query变成glimpse_Q，(batch_size, num_steps, embedding_dim) => (batch_size, num_steps, 1, n_heads, head_dim)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        # 先将glimpse_K后两个维度调换，变为(n_heads, batch_size, num_steps, head_dim, graph_size)，
        # 然后glimpse_Q (n_head, batch_size, num_steps, 1, head_dim)与其做矩阵乘法
        # 得到query的多头注意力值向量compatibility (n_head, batch_size, num_steps, 1, graph_size)
        # 即QK^T/sqrt(d_k)的多头注意力值向量（Q与K的相似度）
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            # 转换掩码形式，mask (batch_size, num_steps, graph_size) => (1, batch_size, num_steps, 1, graph_size)
            # 扩展掩码形式，(1, batch_size, num_steps, 1, graph_size) => (n_heads, batch_size, num_steps, 1, graph_size)
            # 将query的多头注意力值向量进行掩码操作：compatibility中mask为1的位置设置为负无穷（Q与K的掩码相似度）
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        # 将compatibility (n_head, batch_size, num_steps, 1, graph_size)沿最后一个维度求softmax，
        # 然后与glimpse_V (n_head, batch_size, num_steps, graph_size, head_dim)做矩阵乘法
        # 得到query的多头注意力值向量的概率与glimpse_V相乘的多头嵌入向量heads (n_head, batch_size, num_steps, 1, head_dim)
        # 即softmax(QK^T/sqrt(d_k)) * V = Attention(Q, K, V)的多头注意力值向量
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        # 将heads (n_head, batch_size, num_steps, 1, head_dim)转换为(batch_size, num_steps, 1, n_heads, head_dim)
        # 然后将其reshape为(batch_size, num_steps, 1, n_heads * head_dim)
        # 最后通过project_out线性变换，将其映射为(batch_size, num_steps, 1, embedding_dim)
        # 即Attention(Q, K, V)的掩码注意力值向量
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse # query的掩码注意力值向量
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        # 将logit_K (batch_size, 1, graph_size, embedding_dim)转换为(batch_size, 1, embedding_dim, graph_size)
        # 将final_Q (batch_size, num_steps, 1, embedding_dim)与logit_K (batch_size, 1, embedding_dim, graph_size)做矩阵乘法
        # 最后做归一化，得到logits (batch_size, num_steps, 1, graph_size)，然后压缩成(batch_size, num_steps, graph_size)
        # 即QK^T/sqrt(d_k)的注意力值向量
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        # logits即query的注意力值向量与logit_k（编码器输出的glimpse_key的另一种形式）的相似度，形状为(batch_size, num_steps, graph_size)
        # glimpse.squeeze(-1)即query的注意力值向量，形状为(batch_size, num_steps, embedding_dim)
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        """
        功能：将节点的嵌入向量转换为多头注意力机制的形式
        :param v: (batch_size, 1, graph_size, embedding_dim)
        :param num_steps: number of steps to expand to
        :return: (n_heads, batch_size, num_steps, graph_size, head_dim)
        """
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1) # (batch_size, 1, graph_size=21, embedding_dim) => (batch_size, 1, graph_size=21, n_heads, head_dim)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1) # (batch_size, num_steps=1, graph_size=21, n_heads, head_dim)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps=1, graph_size=21, head_dim)
        )
