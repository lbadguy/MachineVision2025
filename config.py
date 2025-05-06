# 交通流量智能分析系统配置文件

# 视频源配置
VIDEO_SOURCE = 0  # 0表示默认摄像头，也可以是视频文件路径

# 背景减除器参数
BG_HISTORY = 200  # 背景模型历史帧数
BG_THRESHOLD = 25  # 背景模型阈值
DETECT_SHADOWS = True  # 是否检测阴影

# 形态学处理参数（参考互联网资源优化）
MORPH_KERNEL_SIZE = (5, 5)  # 形态学处理核大小
MORPH_KERNEL_TYPE = 0  # 核类型：0=RECT, 1=CROSS, 2=ELLIPSE
DILATE_ITERATIONS = 3  # 膨胀操作迭代次数
CLOSE_ITERATIONS = 2  # 闭操作迭代次数

# 车辆检测参数
MIN_CONTOUR_AREA = 500  # 最小车辆轮廓面积
MAX_CONTOUR_AREA = 20000  # 最大车辆轮廓面积

# 车辆跟踪参数
MATCHING_THRESHOLD = 50  # 车辆匹配距离阈值
TRACK_TIMEOUT = 1.0  # 车辆跟踪超时时间（秒）

# 交通分析参数
CONGESTION_VEHICLE_COUNT = 5  # 拥堵车辆数阈值
CONGESTION_SPEED_THRESHOLD = 10  # 拥堵速度阈值（像素/帧）
RECORD_INTERVAL = 300  # 数据记录间隔（秒）

# 可视化参数
SHOW_BOUNDING_BOXES = True  # 是否显示车辆边界框
SHOW_VEHICLE_ID = True  # 是否显示车辆ID
SHOW_VEHICLE_SPEED = True  # 是否显示车辆速度
SHOW_HEATMAP = True  # 是否显示热力图
HEATMAP_ALPHA = 0.3  # 热力图透明度

# 区域定义（可选）
# 格式: [(x1,y1), (x2,y2), ...]
ROI = None

# 车型颜色映射
VEHICLE_COLORS = {
    'car': (0, 255, 0),  # 绿色
    'truck': (0, 0, 255),  # 红色
    'bus': (255, 0, 0),  # 蓝色
    'motorcycle': (255, 255, 0),  # 青色
    'unknown': (255, 0, 255)  # 紫色
}

# 数据保存配置
SAVE_DATA = True  # 是否保存交通数据
DATA_DIR = 'traffic_data'  # 数据保存目录