import math
import random
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def buffon_needle_simulation(num_trials, needle_length=1, board_width=2):
    """
    Buffon 投针实验模拟函数
    :param num_trials: 实验次数
    :param needle_length: 针长度（默认 1）
    :param board_width: 平行线间距（默认 2）
    :return: pi_estimate（π 估计值）, intersect_count（相交次数）
    """
    intersect_count = 0
    for _ in range(num_trials):
        # 随机生成针中点到最近平行线的距离（0 ~ board_width/2）
        x = random.uniform(0, board_width / 2)
        # 随机生成针与平行线的夹角（0 ~ π/2 弧度）
        theta = random.uniform(0, math.pi / 2)
        # 判断是否相交
        if x <= (needle_length / 2) * math.sin(theta):
            intersect_count += 1
    # 公式推导：π ≈ (2 * 针长 * 实验次数) / (间距 * 相交次数)
    try:
        pi_estimate = (2 * needle_length * num_trials) / (board_width * intersect_count)
    except ZeroDivisionError:
        # 处理极端情况（理论上不会发生，因实验次数足够大）
        pi_estimate = float('nan')
    return pi_estimate, intersect_count

def main():
    # 实验次数列表（增加次数以提高稳定性）
    trial_numbers = [100, 1000, 10000, 100000, 1000000]
    results = []  # 存储 (实验次数, π估计值)
    
    # 为提高稳定性，对每个实验次数重复多次取平均
    repeat_times = 10
    print("实验次数\t平均π估计值\t标准差\t\t相对误差")
    
    for n in trial_numbers:
        estimates = []
        for _ in range(repeat_times):
            pi_est, _ = buffon_needle_simulation(n)
            estimates.append(pi_est)
        
        # 计算统计量
        mean_estimate = np.mean(estimates)
        std_dev = np.std(estimates)
        relative_error = abs(mean_estimate - math.pi) / math.pi * 100
        
        results.append((n, mean_estimate))
        print(f"{n}\t\t{mean_estimate:.6f}\t\t{std_dev:.6f}\t\t{relative_error:.2f}%")
    
    # 结果可视化（折线图）
    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in results], [r[1] for r in results], marker='o', linestyle='-', color='b', label='实验估计值')
    plt.axhline(y=math.pi, color='r', linestyle='--', label='真实 π 值')
    
    # 添加误差条（展示标准差）
    yerr = [np.std([est for _, est in results if _ == n]) for n, _ in results]
    plt.errorbar([r[0] for r in results], [r[1] for r in results], yerr=yerr, fmt='none', ecolor='gray', capsize=5)
    
    plt.xscale('log')  # 对数横轴，方便观察趋势
    plt.xlabel('实验次数（对数刻度）')
    plt.ylabel('π 估计值')
    plt.title('Buffon 投针实验：实验次数与 π 估计值关系')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # 添加注释说明实验特性
    plt.annotate(f'重复实验次数: {repeat_times}次', 
                xy=(0.05, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('buffon_needle_result.png', dpi=300)  # 保存高清图表
    plt.show()

if __name__ == "__main__":
    main()
