import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.widgets import Slider, Button, RadioButtons

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class HydrogenAtomSimulation:
    def __init__(self):
        # 物理常数
        self.a = 5.29e-2  # 玻尔半径 (nm)
        self.D_max = 1.1  # 最大概率密度
        self.r0 = 0.25  # 收敛半径 (nm)

        # 模拟参数
        self.num_points = 10000  # 采样点数
        self.r_max = 0.5  # 最大半径 (nm)
        self.n_quantum = 1  # 主量子数

        # 创建图形
        self.fig = plt.figure(figsize=(15, 6))
        self.fig.subplots_adjust(bottom=0.25)

        # 3D电子云图
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title('氢原子电子云模拟 (n=1, l=0, m=0)')
        self.ax1.set_xlabel('X (nm)')
        self.ax1.set_ylabel('Y (nm)')
        self.ax1.set_zlabel('Z (nm)')
        self.ax1.set_xlim([-self.r_max, self.r_max])
        self.ax1.set_ylim([-self.r_max, self.r_max])
        self.ax1.set_zlim([-self.r_max, self.r_max])

        # 径向概率分布图
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('径向概率分布')
        self.ax2.set_xlabel('半径 (nm)')
        self.ax2.set_ylabel('概率密度')
        self.ax2.set_xlim([0, self.r_max])
        self.ax2.set_ylim([0, self.D_max * 1.1])

        # 添加参数滑块
        self.ax_num_points = self.fig.add_axes([0.25, 0.15, 0.65, 0.03])
        self.ax_r_max = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.ax_n_quantum = self.fig.add_axes([0.25, 0.05, 0.65, 0.03])

        self.slider_num_points = Slider(self.ax_num_points, '采样点数', 1000, 50000, valinit=self.num_points,
                                        valstep=1000)
        self.slider_r_max = Slider(self.ax_r_max, '最大半径 (nm)', 0.1, 1.0, valinit=self.r_max, valstep=0.05)
        self.slider_n_quantum = Slider(self.ax_n_quantum, '主量子数 (n)', 1, 3, valinit=self.n_quantum, valstep=1,
                                       valfmt='%d')

        # 绑定滑块事件
        self.slider_num_points.on_changed(self.update)
        self.slider_r_max.on_changed(self.update)
        self.slider_n_quantum.on_changed(self.update)

        # 初始化模拟
        self.update(None)

    def probability_density(self, r, n=1):
        """计算氢原子基态的概率密度函数 D(r)"""
        if n == 1:
            # 基态 (n=1, l=0, m=0)
            return (4 * r ** 2 / self.a ** 3) * np.exp(-2 * r / self.a)
        elif n == 2:
            # 第一激发态 (n=2, l=0, m=0)
            return (r ** 2 / (32 * self.a ** 3)) * np.exp(-r / self.a) * (2 - r / self.a) ** 2
        elif n == 3:
            # 第二激发态 (n=3, l=0, m=0)
            return (4 * r ** 2 / (243 * 81 * self.a ** 3)) * np.exp(-2 * r / (3 * self.a)) * (
                        27 - 18 * r / self.a + 2 * r ** 2 / self.a ** 2) ** 2
        else:
            raise ValueError("仅支持 n=1,2,3")

    def generate_points(self):
        """根据概率密度函数生成电子位置"""
        n = int(self.slider_n_quantum.val)
        points = []
        colors = []

        # 找到概率密度的最大值，用于归一化颜色
        r_values = np.linspace(0, self.slider_r_max.val, 1000)
        d_values = self.probability_density(r_values, n)
        max_density = np.max(d_values)

        while len(points) < int(self.slider_num_points.val):
            # 在球体体积内均匀采样
            x = np.random.uniform(-self.slider_r_max.val, self.slider_r_max.val)
            y = np.random.uniform(-self.slider_r_max.val, self.slider_r_max.val)
            z = np.random.uniform(-self.slider_r_max.val, self.slider_r_max.val)

            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            if r > self.slider_r_max.val:
                continue

            # 计算该位置的概率密度
            density = self.probability_density(r, n)

            # 接受-拒绝采样
            prob = density / max_density
            if np.random.random() < prob:
                points.append([x, y, z])
                # 颜色根据密度归一化
                colors.append(density / max_density)

        return np.array(points), np.array(colors)

    def update(self, val):
        """更新模拟"""
        self.ax1.clear()
        self.ax2.clear()

        # 获取当前参数值
        self.num_points = int(self.slider_num_points.val)
        self.r_max = self.slider_r_max.val
        self.n_quantum = int(self.slider_n_quantum.val)

        # 设置标题显示当前量子数
        self.ax1.set_title(f'氢原子电子云模拟 (n={self.n_quantum}, l=0, m=0)')

        # 生成电子位置
        points, point_colors = self.generate_points()

        # 绘制3D电子云图
        scatter = self.ax1.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=point_colors, cmap='Blues', alpha=0.6, s=2,
            norm=colors.Normalize(vmin=0, vmax=1)
        )
        self.ax1.set_xlim([-self.r_max, self.r_max])
        self.ax1.set_ylim([-self.r_max, self.r_max])
        self.ax1.set_zlim([-self.r_max, self.r_max])
        self.ax1.set_xlabel('X (nm)')
        self.ax1.set_ylabel('Y (nm)')
        self.ax1.set_zlabel('Z (nm)')

        # 添加颜色条
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.fig.colorbar(scatter, ax=self.ax1, pad=0.1)
            self.colorbar.set_label('相对概率密度')
        else:
            self.colorbar.update_normal(scatter)

        # 绘制径向概率分布图
        r_values = np.linspace(0, self.r_max, 1000)
        d_values = self.probability_density(r_values, self.n_quantum)

        self.ax2.plot(r_values, d_values, 'b-', linewidth=2)
        self.ax2.set_xlim([0, self.r_max])
        self.ax2.set_ylim([0, np.max(d_values) * 1.1])
        self.ax2.set_xlabel('半径 (nm)')
        self.ax2.set_ylabel('概率密度')
        self.ax2.set_title('径向概率分布')

        # 标记最大概率位置
        max_idx = np.argmax(d_values)
        r_max_prob = r_values[max_idx]
        self.ax2.axvline(x=r_max_prob, color='r', linestyle='--', alpha=0.7)
        self.ax2.annotate(f'最大概率: r={r_max_prob:.4f} nm',
                          xy=(r_max_prob, d_values[max_idx]),
                          xytext=(r_max_prob + 0.05, d_values[max_idx] * 0.9),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

        # 分析不同参数对电子云分布的影响
        self.analyze_parameter_effects()

        plt.draw()

    def analyze_parameter_effects(self):
        """分析不同参数对电子云分布的影响"""
        # 创建新的图形
        if hasattr(self, 'analysis_fig'):
            plt.close(self.analysis_fig)

        self.analysis_fig = plt.figure(figsize=(15, 10))
        self.analysis_fig.suptitle('不同参数对电子云分布的影响', fontsize=16)

        # 1. 径向距离的影响
        ax1 = self.analysis_fig.add_subplot(221)
        r_values = np.linspace(0, self.r_max, 1000)

        for n in [1, 2, 3]:
            d_values = self.probability_density(r_values, n)
            ax1.plot(r_values, d_values, label=f'n={n}')

        ax1.set_xlabel('半径 (nm)')
        ax1.set_ylabel('概率密度')
        ax1.set_title('不同主量子数的径向概率分布')
        ax1.legend()
        ax1.grid(True)

        # 2. 概率密度分布的3D可视化
        ax2 = self.analysis_fig.add_subplot(222, projection='3d')

        # 创建3D网格
        x = np.linspace(-self.r_max, self.r_max, 50)
        y = np.linspace(-self.r_max, self.r_max, 50)
        z = np.linspace(-self.r_max, self.r_max, 50)
        X, Y, Z = np.meshgrid(x, y, z)

        # 计算每个点的概率密度
        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        D = self.probability_density(R, self.n_quantum)

        # 只显示概率密度较大的点以提高性能
        threshold = np.max(D) * 0.1
        mask = D > threshold

        # 为了可视化，选择一部分点
        step = 2
        X_masked = X[mask][::step]
        Y_masked = Y[mask][::step]
        Z_masked = Z[mask][::step]
        D_masked = D[mask][::step]

        # 绘制3D密度图
        scatter = ax2.scatter(
            X_masked, Y_masked, Z_masked,
            c=D_masked, cmap='Blues', alpha=0.5, s=2,
            norm=colors.Normalize(vmin=0, vmax=np.max(D))
        )

        ax2.set_xlabel('X (nm)')
        ax2.set_ylabel('Y (nm)')
        ax2.set_zlabel('Z (nm)')
        ax2.set_title(f'电子云概率密度分布 (n={self.n_quantum})')
        self.analysis_fig.colorbar(scatter, ax=ax2, pad=0.1, label='概率密度')

        # 3. 不同参数对电子云扩展的影响
        ax3 = self.analysis_fig.add_subplot(223)

        # 计算不同半径处的累积概率
        r_values = np.linspace(0, self.r_max, 1000)
        for n in [1, 2, 3]:
            d_values = self.probability_density(r_values, n)
            # 数值积分计算累积概率
            cumulative_prob = np.cumsum(d_values) * (r_values[1] - r_values[0])
            cumulative_prob /= cumulative_prob[-1]  # 归一化

            ax3.plot(r_values, cumulative_prob, label=f'n={n}')

        ax3.set_xlabel('半径 (nm)')
        ax3.set_ylabel('累积概率')
        ax3.set_title('不同主量子数的电子云累积概率分布')
        ax3.legend()
        ax3.grid(True)

        # 4. 概率密度函数与玻尔模型的比较
        ax4 = self.analysis_fig.add_subplot(224)

        # 玻尔半径
        a0_values = [self.a * n ** 2 for n in [1, 2, 3]]

        # 绘制氢原子基态的概率密度
        r_values = np.linspace(0, self.r_max, 1000)
        d_values_ground = self.probability_density(r_values, 1)
        ax4.plot(r_values, d_values_ground, 'b-', label='基态 (n=1)')

        # 标记玻尔半径位置
        for i, a0 in enumerate(a0_values):
            ax4.axvline(x=a0, color=f'C{i + 1}', linestyle='--', alpha=0.7)
            ax4.annotate(f'玻尔半径 n={i + 1}: {a0:.4f} nm',
                         xy=(a0, 0),
                         xytext=(a0 + 0.02, 0.1 + i * 0.2),
                         rotation=90)

        ax4.set_xlabel('半径 (nm)')
        ax4.set_ylabel('概率密度')
        ax4.set_title('概率密度函数与玻尔模型的比较')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
        plt.show()

    def run(self):
        """运行模拟"""
        plt.show()


if __name__ == "__main__":
    simulation = HydrogenAtomSimulation()
    simulation.run()
