# 导入 OR-Tools 库
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import torch
import time
import logging
import datetime
import argparse

import sys
import os
# 添加上级目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import torch_load_cpu, load_problem

DISTANCE_TIME_SCALE = 1000  # 距离和速度的缩放比例
WORK_TIME = 8

def create_data_model(data_tensor, penalty, capacity_scale):
    PENALTY = penalty # 每小时的违反时间窗的惩罚
    SPEED = 2
    CAPACITY = 1
    VEHICLES = 12
    """
    创建CVRPTW的数据
    :param data: 包含CVRPTW数据的字典（张量）
    :return: 包含CVRPTW数据的字典（列表）
    """
    depot_tensor = data_tensor['depot']
    loc_tensor = data_tensor['loc']
    loc_demand_tensor = data_tensor['demand']
    loc_tw_tensor = data_tensor['tw']

    # 拼接depot_tensor和data['loc']，得到所有节点的坐标
    depot_loc_tensor = torch.cat([depot_tensor.unsqueeze(0), loc_tensor], dim=0)
    depot_loc_demand_tensor = torch.cat([torch.tensor([0]), loc_demand_tensor], dim=0)
    depot_loc_tw_tensor = torch.cat([torch.tensor([[0, 1]]), loc_tw_tensor], dim=0)

    # 计算节点之间的欧几里得距离
    distance_matrix_tensor = compute_euclidean_distance_matrix(depot_loc_tensor)

    data = {}
    data['distance_matrix'] = (distance_matrix_tensor.numpy() * DISTANCE_TIME_SCALE).tolist()  # 根据节点和仓库坐标计算
    data['vehicle_speed'] = SPEED  # 车辆的速度，单位KM/h
    data['time_matrix'] = (distance_matrix_tensor / data['vehicle_speed'] * DISTANCE_TIME_SCALE).numpy().tolist()      # 根据节点和仓库坐标及车辆速度计算
    data['time_windows'] = (depot_loc_tw_tensor * DISTANCE_TIME_SCALE * WORK_TIME).numpy().tolist()     # 节点的时间窗
    data['demands'] = (depot_loc_demand_tensor*CAPACITY*capacity_scale).numpy().tolist()       # 节点的需求量
    data['num_vehicles'] = VEHICLES         # 车辆的数量
    data['vehicle_capacities'] = [CAPACITY*capacity_scale] * VEHICLES  # 车辆的容量
    data['penalty'] = PENALTY  # 违反时间窗的惩罚系数
    data['depot'] = 0           # 仓库位置的索引
    return data

def distance_callback(from_index, to_index):
    """
    创建和注册距离回调
    :param from_index: 起始节点的索引
    :param to_index: 终止节点的索引
    :return: 从起始节点到终止节点的距离
    """
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

def demand_callback(from_index):
    """
    节点需求量的回调函数
    :param from_index: 节点的索引
    :return: 节点的需求量
    """
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]

def time_callback(from_index, to_index):
    """
    创建节点转移的时间回调函数
    :param from_index: 起始节点的索引
    :param to_index: 终止节点的索引
    :return: 从起始节点到终止节点的时间
    """
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['time_matrix'][from_node][to_node]

def load_dataset(node_size, filename, load_size=512, offset=0):
    """
    加载数据集
    :param index: 数据集的索引
    :return: 数据集中的第index组数据
    """
    problem = load_problem('cvrptw')
    val_dataset = problem.make_dataset(  # 加载或生成验证集
        size=node_size,
        num_samples=load_size,
        offset=offset,
        filename=filename,
        distribution='all')
    return val_dataset

def compute_euclidean_distance_matrix(locations):
    """
    根据节点的坐标计算节点之间的欧几里得距离
    :param locations: 节点的坐标（torch二维张量）
    :return: 节点之间的欧几里得距离矩阵（torch二维张量）
    """
    # 展开 locations 以形成两个 [N, 1, 2] 和 [1, N, 2] 的张量
    locations_expanded = locations.unsqueeze(1)  # Shape [N, 1, 2]
    locations_t_expanded = locations.unsqueeze(0)  # Shape [1, N, 2]

    distances_squared = torch.pow(locations_expanded - locations_t_expanded, 2).sum(2) # 利用广播计算所有配对的平方差
    distance_matrix = torch.sqrt(distances_squared) # 计算欧几里得距离，即取平方根

    return distance_matrix

def get_solution(data, manager, routing, solution, data_index, capacity_scale):
    """
    打印解决方案
    :param data: 包含CVRPTW数据的字典
    :param manager: 路由索引管理器，用于转换节点和索引
    :param routing: 路由模型，用于获取解决方案
    :param solution: 问题的解决方案
    """
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            time_var = routing.GetDimensionOrDie('Time').CumulVar(index)
            route_distance += routing.GetArcCostForVehicle(index, next_index, vehicle_id)
            route_load += data['demands'][node_index]
            plan_output += ' {}({},{})[{},{}] ->'.format(node_index,
                                                      data['time_windows'][node_index][0]/DISTANCE_TIME_SCALE,
                                                      data['time_windows'][node_index][1]/DISTANCE_TIME_SCALE,
                                                      solution.Min(time_var)/DISTANCE_TIME_SCALE,
                                                      solution.Max(time_var)/DISTANCE_TIME_SCALE)
            index = next_index
        node_index = manager.IndexToNode(index)
        total_distance += route_distance
        total_load += route_load
        plan_output += ' {0} Load({1})\n'.format(node_index, route_load/capacity_scale)
        plan_output += 'Distance of the route: {}km\n'.format(route_distance/DISTANCE_TIME_SCALE)
        plan_output += 'Load of the route: {}\n'.format(route_load/capacity_scale)
        if data_index % 50 == 0:
            print(plan_output)
    if data_index % 50 == 0:
        print('Total Distance of all routes: {}'.format(total_distance/DISTANCE_TIME_SCALE))
        print('Total Load of all routes: {}'.format(total_load/capacity_scale))
    # 打印目标函数值
    objective_value_real = solution.ObjectiveValue()/DISTANCE_TIME_SCALE
    total_distance_real = total_distance/DISTANCE_TIME_SCALE
    total_load_real = total_load/capacity_scale
    # print('Objective: {}'.format(objective_value_real))
    return objective_value_real, total_distance_real, total_load_real

def setup_logging():
    """
    设置日志记录
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器，格式为cvrptw-日期.log
    date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_handler = logging.FileHandler(f'cvrptw-{date_str}.log')
    file_handler.setLevel(logging.INFO)
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


if __name__ == '__main__':
    # 设置日志记录
    setup_logging()

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Run OR-Tools CVRPTW')
    parser.add_argument('--load_size', type=int, default=10000, help='Number of instances to load')
    parser.add_argument('--load_file', type=str, default='../data/cvrptw/cvrptw100_test_seed1234.pkl', help='File to load instances from')
    parser.add_argument('--capacity_scale', type=int, default=50, help='Capacity of the vehicles')
    parser.add_argument('--penalty', type=int, default=2, help='Penalty for violating time windows')

    # 解析命令行参数
    args = parser.parse_args()
    load_size = args.load_size
    load_file = args.load_file
    capacity_scale = args.capacity_scale
    penalty = args.penalty

    # 记录时间
    start_time = time.time()

    offset=0
    objective_value_list = []
    distance_list = []
    violation_times_list = []
    violation_customers_list = []
    # filename = '../data/cvrptw_20_train.pkl'
    # filename='../data/cvrptw/cvrptw20_test_seed1234.pkl'
    # filename='../data/cvrptw/cvrptw50_test_seed1234.pkl'
    # filename='../data/cvrptw/cvrptw100_test_seed1234.pkl'
    # filename='../data/cvrptw/cvrptw20_validation_seed4321.pkl'
    logging.info(f'dataset: {load_file}, load_size: {load_size}, capacity_scale: {capacity_scale}, penalty: {penalty}')

    # 加载数据集
    data_tensor = load_dataset(node_size=20, filename=load_file, load_size=load_size, offset=offset)

    for data_index in range(load_size):
        # 初始化数据
        data = create_data_model(data_tensor[data_index], penalty, capacity_scale)

        # 创建路由索引管理器和路由模型
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        """
        注册节点转移的回调函数
        """
        distance_callback_index = routing.RegisterTransitCallback(distance_callback) # 注册节点转移的距离回调函数
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)  # 注册节点需求量的回调函数
        time_callback_index = routing.RegisterTransitCallback(time_callback)  # 注册节点转移的时间回调函数

        """
        定义每个边的成本，即行驶距离
        """
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        # routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

        """
        添加约束
        """
        # 添加车辆容量约束
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # 没有容量松弛
            data['vehicle_capacities'],  # 车辆最大容量限制（数组）
            True,  # 从0开始累积
            'Capacity'
        )

        # 添加时间窗约束
        routing.AddDimension(
            time_callback_index,
            8 * DISTANCE_TIME_SCALE,  # 等待时间松弛量（时间窗内）
            8 * DISTANCE_TIME_SCALE,  # 最大的时间限制
            True,  # 不要强制开始累积为零
            'Time'
        )
        # 为每个节点（包括仓库）添加时间窗
        time_dimension = routing.GetDimensionOrDie('Time') # 获取时间维度
        for i, time_window in enumerate(data['time_windows']):
            index = manager.NodeToIndex(i)
            # 设置节点的时间窗（硬限制）
            # time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
            # 设置节点的时间窗（软限制）
            time_dimension.SetCumulVarSoftUpperBound(index, int(time_window[1]), int(data['penalty']))
            time_dimension.SetCumulVarSoftLowerBound(index, int(time_window[0]), int(data['penalty']))

        """
        设置搜索参数
        """
        # 最小化车辆的总行驶距离
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        """
        求解问题
        """

        # print('第{}个数据'.format(data_index))
        logging.info(f'第{data_index}个数据')
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            objective_value, distance, _ = get_solution(data, manager, routing, solution, data_index, capacity_scale)
            objective_value_list.append(objective_value)
            distance_list.append(distance)
            violation_times = 0
            violated_customers = 0
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index !=data['depot']:
                        route_time = solution.Min(time_dimension.CumulVar(index))
                        start, end = data['time_windows'][node_index]
                        if route_time < start:
                            violation_time = start - route_time
                            violation_times += violation_time
                            violated_customers += 1
                        elif route_time > end:
                            violation_time = route_time - end
                            violation_times += violation_time
                            violated_customers += 1
                    index = solution.Value(routing.NextVar(index))
            violation_times_list.append(violation_times)
            violation_customers_list.append(violated_customers)

        else:
            print('No solution found !')

        # 打印目标函数值
        logging.info(f'Objective Value: {objective_value}')
        # 打印总行驶距离
        logging.info(f'Total Distance: {distance}')
        # 打印违反的时间窗数量
        logging.info(f'Total violation time: {violation_times/DISTANCE_TIME_SCALE} hours')
        # 打印违反的时间窗约束的客户数量
        logging.info(f'Violated customers: {violated_customers}')

    # 打印平均目标函数值
    logging.info('\n')
    logging.info(f'Mean values of all instances')
    logging.info(f'Mean Objective Value: {sum(objective_value_list) / len(objective_value_list)}')
    # 打印平均总行驶距离
    logging.info(f'Mean Total Distance: {sum(distance_list) / len(distance_list)}')
    # 打印违背的时间窗数量
    logging.info(f'Mean violation time: {sum(violation_times_list) / DISTANCE_TIME_SCALE / len(violation_times_list)} hours')
    # 打印违背的时间窗约束的客户数量
    logging.info(f'Mean violated customers: {sum(violation_customers_list) / len(violation_customers_list)}')
    # 打印时间
    logging.info(f'Caculated Time: {time.time() - start_time} seconds')


