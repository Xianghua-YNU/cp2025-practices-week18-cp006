import random
import matplotlib.pyplot as plt

def simulate_chain_reaction(initial_neutrons=1, generations=20, k=2.5, survival_prob=0.8):
    """
    模拟链式反应过程
    :param initial_neutrons: 初始中子数
    :param generations: 模拟代数
    :param k: 平均每次裂变产生的中子数（实际取整后随机）
    :param survival_prob: 中子存活并引发裂变的概率
    :return: 每代中子数列表
    """
    neutron_counts = [initial_neutrons]  # 记录每代中子数
    for _ in range(generations):
        current_neutrons = neutron_counts[-1]
        next_neutrons = 0
        
        for _ in range(current_neutrons):
            # 模拟中子存活概率
            if random.random() < survival_prob:
                # 随机生成裂变产生的中子数（取整处理）
                emitted_neutrons = random.randint(1, int(k) + 1)  # 允许一定波动
                next_neutrons += emitted_neutrons
        
        neutron_counts.append(next_neutrons)
    return neutron_counts

def analyze_parameters():
    """
    分析不同参数对链式反应的影响
    """
    params = [
        {"label": "k=1.5 (次临界)", "k": 1.5, "survival_prob": 0.7},
        {"label": "k=1.0 (临界)", "k": 1.0, "survival_prob": 0.6},
        {"label": "k=2.5 (超临界)", "k": 2.5, "survival_prob": 0.8}
    ]
    
    plt.figure(figsize=(10, 6))
    for p in params:
        counts = simulate_chain_reaction(
            initial_neutrons=1,
            generations=15,
            k=p["k"],
            survival_prob=p["survival_prob"]
        )
        plt.plot(range(len(counts)), counts, marker='o', label=p["label"])
    
    plt.xlabel("代数")
    plt.ylabel("中子数")
    plt.title("不同增殖系数(k)对链式反应的影响")
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行模拟
if __name__ == "__main__":
    # 示例：超临界状态模拟
    result = simulate_chain_reaction(
        initial_neutrons=1,
        generations=10,
        k=2.8,       # 增殖系数（>1时反应持续）
        survival_prob=0.9  # 中子存活概率
    )
    print("各代中子数：", result)
    
    # 绘制参数分析图
    analyze_parameters()
