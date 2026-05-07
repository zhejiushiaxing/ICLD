import matplotlib.pyplot as plt
import numpy as np
from typing import Optional  # 类型注解，提升代码可读性
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from matplotlib.colors import ListedColormap

def plot_01_matrix(matrix: list[list], title: str = "per_layer_logits", save_path: Optional[str] = None) -> None:
    """
    可视化01矩阵（输入为list[list]格式），支持指定路径保存图片
    :param matrix: 待可视化的01矩阵，要求每行长度一致，元素仅为0或1
    :param title: 可视化图像的标题，默认值为"per_layer_logits"（英文标题避免中文字体警告）
    :param save_path: 图片保存路径（含目录+文件名+后缀），可选参数，默认None（不保存）
    :raises ValueError: 若矩阵为空、行列长度不一致、包含非0/1元素时抛出异常
    :raises IOError: 若保存路径无效（如目录不存在、无写入权限）时抛出异常
    """
    # 步骤1：输入合法性校验
    if not matrix or not matrix[0]:
        raise ValueError("输入的01矩阵不能为空，且至少包含1行1列")
    row_len = len(matrix[0])
    for row in matrix:
        if len(row) != row_len:
            raise ValueError("输入的01矩阵所有行的列数必须一致")
        for num in row:
            if num not in (0, 1):
                raise ValueError("01矩阵的元素必须仅包含0和1（整数/浮点数均可）")
    
    # 步骤2：转换为numpy数组
    mat_np = np.array(matrix, dtype=int)
    
    # 步骤3：自定义配色（0→白色，1→绿色）
    cmap = ListedColormap(['#FFFFFF', '#34A853'])  # 白色+谷歌绿，视觉更舒适
    
    # 步骤4：初始化画布
    plt.figure(figsize=(8, 6))
    # 绘制矩阵：自定义配色+正方形单元格
    plt.imshow(mat_np, cmap=cmap, aspect='equal')
    
    # 步骤5：田字格网格线（黑色粗线，清晰划分每个单元格）
    plt.grid(
        color='black',       # 黑色网格线
        linewidth=1.5,       # 加粗线条，增强田字格效果
        which='both',        # 同时显示主/次网格线
        axis='both',         # 同时显示x/y轴网格线
        linestyle='-'        # 实线样式
    )
    
    # 步骤6：图像美化配置
    plt.title(title, fontsize=14, pad=20, fontweight='bold')  # 加粗标题更醒目
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
    plt.tight_layout()  # 自动调整布局
    
    # 步骤7：保存图片到指定目录
    if save_path is not None:
        try:
            plt.savefig(
                save_path,
                bbox_inches='tight',
                dpi=300,
                facecolor='white'
            )
            print(f"✅ 01矩阵可视化图片已成功保存至：{save_path}")
        except IOError as e:
            raise IOError(f"❌ 图片保存失败！请检查路径是否有效（目录需已存在、有写入权限），错误信息：{str(e)}")
    
    plt.show()
    plt.close()




# ------------------- 测试示例：直接运行即可体验所有功能 -------------------
if __name__ == "__main__":
    # 测试用01矩阵（可替换为你的自定义矩阵）
    test_matrix = [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1]
    ]
    
    # 示例1：仅可视化，不保存图片（兼容原使用方式）
    # plot_01_matrix(test_matrix, title="仅可视化-不保存")
    
    # 示例2：保存图片到【当前目录】（直接写文件名+后缀）
    # plot_01_matrix(test_matrix, title="保存至当前目录", save_path="01_matrix.png")
    
    # 示例3：保存图片到【自定义子目录】（推荐，需确保目录已存在！）
    plot_01_matrix(
        matrix=test_matrix,
        title="01矩阵可视化（保存至指定目录）",
        save_path="/mnt/Data/zhoujiaxing/code/ICLD/demo/01_matrix_visual.jpg"  # 目录层级可自定义
    )