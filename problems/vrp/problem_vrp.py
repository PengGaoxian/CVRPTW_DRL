from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np

from problems.vrp.state_cvrp import StateCVRP, StateCVRPTW
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class CVRPTW(object):

    NAME = 'cvrptw'  # Capacitated Vehicle Routing Problem with Time Windows

    VEHICLE_CAPACITY = 1.0  # 车辆容量为1，demands需要被缩放

    WORK_TIME = 8.0  # 一天工作8小时
    TIME_WINDOW_LENGTH = 2  # 时间窗的长度为2小时，即一天被分为4个时间窗分别是9:00-11:00, 11:00-13:00, 13:00-15:00, 15:00-17:00，客户的时间窗是从这4个时间窗中选择的

    VEHICLE_SPEED = 2  # speed of the vehicle 领域边长为1.0，设置车辆速度为1小时1个边长
    PENALTY = WORK_TIME * 2  # 以小时为单位违反时间窗的惩罚系数

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: 字典，包含了问题的所有数据（depot、loc、demand、time windows）
        :param pi: 张量(batch_size, steps)，表示一种可能的路径，即节点的访问顺序
        :return cost: cost = distance + 系数*违背时间窗的客户数量
        :return Mask: None
        """
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0] # 一个batch中的路径都是一样的，所以只需要检查一个路径即可

        """
        验证路径是否合法，即路径中包含节点1到n
        """
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"


        """
        将仓库的demand设置为-VEHICLE_CAPACITY，与其他客户的demand拼接，得到[batch_size, graph_size+1]
        然后按照路径pi，将demand取出，得到[batch_size, steps]
        """
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi) # 按照路径pi，将demand_with_depot中的demand取出，得到[batch_size, steps]


        """
        遍历路径pi中的每个节点，计算used_cap，验证是否超过容量
        """
        used_cap = torch.zeros_like(dataset['demand'][:, 0]) # 初始化已用容量为0 [batch_size]
        # 遍历路径中的每个节点，计算used_cap，验证是否超过容量
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"


        """
        拼接仓库坐标和节点坐标，得到[batch_size, graph_size+1, 2]
        按照路径pi，将坐标取出，得到[batch_size, steps, 2]
        计算路径pi的欧几里得距离，加上从仓库出发到第一个节点，加上从最后一个节点到仓库
        """
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1) # 仓库坐标和节点坐标拼接，得到[batch_size, graph_size+1, 2]
        pi_coords = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))) # 按照路径pi，将loc_with_depot中的坐标取出，得到[batch_size, steps, 2]

        first_to_last = (pi_coords[:, 1:] - pi_coords[:, :-1]).norm(p=2, dim=2).sum(1) # [batch_size] 每个节点到下一个节点的距离之和
        depot_to_first = (pi_coords[:, 0] - dataset['depot']).norm(p=2, dim=1) # [batch_size] 第一个节点到仓库
        last_to_depot = (pi_coords[:, -1] - dataset['depot']).norm(p=2, dim=1) # [batch_size] 最后一个节点到仓库
        route_length = first_to_last + depot_to_first + last_to_depot # [batch_size] 总路径长度

        violated_time_num, violated_node_num, arrival_time = CVRPTW.calculate_time_windows_violations(dataset, pi)

        cost = route_length + CVRPTW.PENALTY * violated_time_num.squeeze()

        return cost, None, violated_time_num.squeeze(), violated_node_num.squeeze(), route_length

    @staticmethod
    def calculate_time_windows_violations(dataset, pi):
        """
        计算违反时间窗的客户数量，以及车辆到达最后一个节点的时间
        :param dataset: 字典，包含了问题的所有数据（depot、loc、demand、tw）
        :param pi: 张量(batch_size, steps)，表示一种可能的路径，即节点的访问顺序
        :return violated_tw_time_num: 违反时间窗的时间
        :return violated_tw_node_num: 违反时间窗的客户数量
        :return arrival_time: 车辆到达最后一个节点的时间
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        depot_tw = (torch.full((dataset['tw'].size(0), 1, 2), fill_value=0)/CVRPTW.WORK_TIME).to(device)  # 仓库的时间窗为[0, 0]/CVRTPTW.WORK_TIME
        depot_tw[:, :, 1] = 8/CVRPTW.WORK_TIME  # 仓库的时间窗为[0, 8]/CVRPTW.WORK_TIME

        depot_loc_tw = torch.cat((depot_tw, dataset['tw']), 1)  # 拼接仓库时间窗和节点时间窗 [batch_size, graph_size+1, 2]

        depot_pi = torch.cat((torch.zeros_like(pi[:, :1]), pi), 1)  # 在路径pi的前面添加仓库节点 [batch_size, steps+1]
        depot_pi_tw = depot_loc_tw.gather(1, depot_pi[..., None].expand(*depot_pi.size(), depot_loc_tw.size(-1)))  # 按照路径depot_pi，将时间窗信息取出 [batch_size, steps+1, 2]

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # 拼接仓库坐标和节点坐标 [batch_size, graph_size+1, 2]
        depot_pi_coords = loc_with_depot.gather(1, depot_pi[..., None].expand(*depot_pi.size(), loc_with_depot.size(-1)))  # 按照路径depot_pi，将坐标取出 [batch_size, steps+1, 2]

        batch_size, steps = depot_pi.shape
        violated_tw_node_num = torch.zeros(batch_size, 1).to(device)  # 初始化违反节点时间窗violated_node_num [batch_size, 1]
        violated_tw_time_num = torch.zeros(batch_size, 1).to(device)  # 初始化违反节点时间窗violated_node_num [batch_size, 1]
        arrival_time = torch.zeros(batch_size, 1).to(device)  # 初始化车辆到达节点的时间arrival_time [batch_size, 1]

        # 遍历depot_pi [batch_size, steps+2]
        for i in range(steps - 1):
            distance = (depot_pi_coords[:, i + 1] - depot_pi_coords[:, i]).norm(p=2, dim=1)[:, None]  # 计算节点之间的欧几里得距离

            current_step_node = depot_pi[:, i + 1].unsqueeze(1)  # 获取当前step所在的节点 [batch_size, 1]

            arrival_time += distance / CVRPTW.VEHICLE_SPEED / CVRPTW.WORK_TIME  # 计算到达下一个节点（i+1）的时间
            arrival_time = torch.where(current_step_node == 0, torch.zeros_like(arrival_time),
                                       arrival_time)  # 如果当前节点是depot，则将arrival_time置为0

            selected_tw = torch.gather(depot_loc_tw, 1, current_step_node[:, :, None].expand(batch_size, 1, 2))  # 当前节点的时间窗 (batch_size, 1, 2)
            """
            如果到达当前节点的时间cur_arrival_time小于时间窗的左边界，则需要等待到时间窗的左边界，即cur_arrival_time = 时间窗的左边界；
            如果cur_arrival_time大于时间窗的右边界，则违反时间窗。
            """
            arrival_time = torch.where(arrival_time < selected_tw[:, :, 0], selected_tw[:, :, 0], arrival_time)
            violated_tw_node_num = torch.where(arrival_time > selected_tw[:, :, 1], violated_tw_node_num + 1, violated_tw_node_num)  # 如果到达节点的时间超过该节点的右时间窗，violated_tw_node_num+1
            violated_tw_time_num = torch.where(arrival_time > selected_tw[:, :, 1], violated_tw_time_num + arrival_time - selected_tw[:, :, 1], violated_tw_time_num)  # 如果到达节点的时间超过该节点的右时间窗，violated_tw_time_num+超出时间

        return violated_tw_time_num * CVRPTW.WORK_TIME, violated_tw_node_num, arrival_time * CVRPTW.WORK_TIME


    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRPTW.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    # 输入dataset：字典，包含了问题的所有数据（depot、loc、demand）
    # 输入pi：张量(batch_size, steps)，表示一种可能的路径，即节点的访问顺序
    # 输出cost：路径的总长度
    # 输出Mask：None
    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }

def make_instance_cvrptw(args):
    depot, loc, demand, capacity, tw, worktime, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'tw': torch.tensor(tw, dtype=torch.float) / worktime
    }

class CVRPTWDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super().__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance_cvrptw(args) for args in data[offset:offset+num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                30: 35.,
                50: 40.,
                70: 45.,
                100: 50.
            }

            """
            生成时间窗数据，形状为[batch_size, 2]，左时间窗从[0,2,4,6]中随机选择，右时间窗为左时间窗+2
            """
            # time_windows_left_options = np.arange(0, CVRPTW.WORK_TIME, CVRPTW.TIME_WINDOW_LENGTH) # [0, 2, 4, 6]
            # time_windows_right_options = time_windows_left_options + CVRPTW.TIME_WINDOW_LENGTH # [2, 4, 6, 8]
            # time_windows = np.stack((time_windows_left_options, time_windows_right_options), axis=-1) # [[0, 2], [2, 4], [4, 6], [6, 8]]
            time_windows_left_options = torch.arange(0, CVRPTW.WORK_TIME, CVRPTW.TIME_WINDOW_LENGTH)  # [0, 2, 4, 6]
            time_windows_right_options = time_windows_left_options + CVRPTW.TIME_WINDOW_LENGTH  # [2, 4, 6, 8]
            time_windows = torch.stack((time_windows_left_options, time_windows_right_options), dim=-1)  # [[0, 2], [2, 4], [4, 6], [6, 8]]

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'tw': torch.gather(time_windows, 0, torch.IntTensor(size, 1).random_(0, time_windows.shape[0]).repeat(1, 2).long()) / CVRPTW.WORK_TIME
                    # 'tw': torch.from_numpy(time_windows[np.random.choice(time_windows.shape[0], size), :]).float() / CVRPTW.WORK_TIME # 随机选择一个时间窗
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]