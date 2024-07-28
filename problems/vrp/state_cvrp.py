import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRPTW(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # [batch_size, graph_size+1, 2] 仓库和顾客的坐标
    demand: torch.Tensor # [batch_size, graph_size] 顾客的需求
    tw: torch.Tensor # [batch_size, graph_size, 2] 顾客的时间窗

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # 行索引，即批量数据(batch_size)的索引

    # State
    prev_a: torch.Tensor # 上一步选择的动作，即要访问的节点
    used_capacity: torch.Tensor # 已使用的容量
    visited_: torch.Tensor  # 记录已访问的节点
    lengths: torch.Tensor # 记录路径长度
    cur_coord: torch.Tensor # 当前坐标
    i: torch.Tensor  # 记录步数
    cur_arrival_time: torch.Tensor # 当前节点的到达时间
    violated_tw_num: torch.Tensor # 违反时间窗的客户数量
    vehicle_speed: torch.Tensor # 车辆的速度

    VEHICLE_CAPACITY = 1.0  # 与problem_vrp中的VEHICLE_CAPACITY相同
    VEHICLE_SPEED = 2  # 按一小时跑2个边长的速度，与problem_vrp中的VEHICLE_SPEED相同

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            cur_arrival_time=self.cur_arrival_time[key],
            violated_tw_num=self.violated_tw_num[key],
            vehicle_speed=self.vehicle_speed[key]
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot'] # (batch_size, 2)
        loc = input['loc'] # (batch_size, 20, 2)
        demand = input['demand'] # (batch_size, 20)
        tw = input['tw'] # (batch_size, 20, 2)

        batch_size, n_loc, _ = loc.size()
        return StateCVRPTW(
            coords=torch.cat((depot[:, None, :], loc), -2), # 将depot转化为(batch_size, 1, 2)，然后与loc拼接成(batch_size, 21, 2)
            demand=demand, # (batch_size, 20)
            tw = tw, # (batch_size, 20, 2)
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension (batch_size, 1)
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device), # (batch_size, 1)
            used_capacity=demand.new_zeros(batch_size, 1), # (batch_size, 1)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                ) # (batch_size, 1, 21)
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device), # (batch_size, 1)
            cur_coord=input['depot'][:, None, :],  # Add step dimension (batch_size, 1, 2)
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            cur_arrival_time = torch.zeros(batch_size, 1, device=loc.device), # (batch_size, 1)
            violated_tw_num = torch.zeros(batch_size, 1, device=loc.device), # (batch_size, 1)
            # 将车辆的速度转化为张量，形状为(batch_size, 1)
            vehicle_speed = torch.full((batch_size, 1), StateCVRPTW.VEHICLE_SPEED, dtype=torch.float, device=loc.device) # (batch_size, 1)
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        """
        更新状态，包括：
        1. 上一个节点选择的动作prev_a（选择的节点）
        2. 已使用容量used_capacity
        3. 已访问的节点visited_
        4. 已行驶的路径长度lengths
        5. 当前节点的坐标cur_coord
        6. 更新步数i
        7. 当前节点的到达时间cur_arrival_time
        8. 违反时间窗的客户数量violated_tw_num
        :param selected: (batch_size, 1)
        :return: StateCVRPTW
        """

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        selected = selected[:, None]  # Add dimension for step
        prev_a = selected # (batch_size, 1)
        n_loc = self.demand.size(-1)  # 客户节点数量

        cur_coord = self.coords[self.ids, selected] # 当前节点的坐标 (batch_size, 1, 2)
        prev_to_cur_length = (cur_coord - self.cur_coord).norm(p=2, dim=-1) # 上一个节点到当前节点的路径长度 (batch_size, 1)
        lengths = self.lengths + prev_to_cur_length  # 行驶到当前节点的路径长度 (batch_dim, 1)

        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)] # 当前节点的需求 (batch_size, 1)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float() # 行驶到当前节点时，已使用的容量 (batch_size, 1)

        cur_arrival_time = self.cur_arrival_time + prev_to_cur_length/self.VEHICLE_SPEED # 当前节点的到达时间 (batch_size, 1)

        selected_tw = self.tw[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)] # 当前节点的时间窗 (batch_size, 1, 2)
        """
        如果到达当前节点的时间cur_arrival_time小于时间窗的左边界，则需要等待到时间窗的左边界，即cur_arrival_time = 时间窗的左边界；
        如果cur_arrival_time大于时间窗的右边界，则违反时间窗。
        """
        cur_arrival_time = torch.where(cur_arrival_time < selected_tw[:, :, 0], selected_tw[:, :, 0], cur_arrival_time)
        violated_tw_num = torch.where(cur_arrival_time > selected_tw[:, :, 1], self.violated_tw_num + 1, self.violated_tw_num)

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1) # 将当前节点记录到已访问的节点visited_中 (batch_size, 1, n_loc + 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1,
            cur_arrival_time=cur_arrival_time, violated_tw_num=violated_tw_num
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        得到一个(batch_size, n_loc + 1)的掩码，其中0表示可行动作，1表示不可行动作
        loc节点已被访问或loc节点的demand加上已使用的容量大于车辆容量时，则该节点不可行
        depot节点不能连续访问，除非所有节点都已访问
        :return: mask：(batch_size, n_loc + 1)
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))
        """
        demand (1024, 20)
        ids (1024, 1)
        demand[ids] (1024,1,20)
        used_capacity (1024, 1)
        """
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY) # (batch_size, 1, n_loc) 已用容量加上该节点的需求会超过车辆容量的节点不可访问
        """
        tw (1024, 20, 2)
        ids (1024, 1)
        tw[ids, :, 1] (1024, 1, 20)
        cur_arrival_time (1024, 1)
        cur_arrival_time[ids] (1024, 1, 1)
        """
        # exceeds_tw = (self.tw[self.ids, :, 1] < self.cur_arrival_time[self.ids]) # (batch_size, 1, n_loc) 时间窗的右边界在cur_arrival_time之前的节点不可访问

        # mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | exceeds_tw # (batch_size, n_loc) 不能访问的节点
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap # (batch_size, n_loc) 不能访问的节点

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions


class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot'] # (batch_size, 2)
        loc = input['loc'] # (batch_size, 20, 2)
        demand = input['demand'] # (batch_size, 20)

        batch_size, n_loc, _ = loc.size()
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2), # 将depot转化为(batch_size, 1, 2)，然后与loc拼接成(batch_size, 21, 2)
            demand=demand, # (batch_size, 20)
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension (batch_size, 1)
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device), # (batch_size, 1)
            used_capacity=demand.new_zeros(batch_size, 1), # (batch_size, 1)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                ) # (batch_size, 1, 21)
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device), # (batch_size, 1)
            cur_coord=input['depot'][:, None, :],  # Add step dimension (batch_size, 1, 2)
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        得到一个(batch_size, n_loc + 1)的掩码，其中0表示可行动作，1表示不可行动作
        loc节点已被访问或loc节点的demand加上已使用的容量大于车辆容量时，则该节点不可行
        depot节点不能连续访问，除非所有节点都已访问
        :return: mask：(batch_size, n_loc + 1)
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions