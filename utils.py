import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from traffic_analyzer import TrafficAnalyzer

def generate_sample_training_data():
    """
    生成示例训练数据用于车型分类
    这只是一个示例，实际应用中应该使用真实标注数据
    
    返回:
        features: 特征矩阵
        labels: 标签向量
    """
    # 模拟不同车型的特征
    # 特征顺序: [面积, 周长, 长宽比, 矩形度, 圆形度]
    
    # 轿车特征 - 中等面积，较小长宽比
    car_features = [
        [5000, 300, 1.5, 0.8, 0.6],
        [4800, 290, 1.6, 0.85, 0.65],
        [5200, 310, 1.4, 0.82, 0.62],
        [4500, 280, 1.5, 0.83, 0.63],
        [5100, 305, 1.55, 0.81, 0.61]
    ]
    
    # 卡车特征 - 大面积，大长宽比
    truck_features = [
        [12000, 500, 2.5, 0.9, 0.4],
        [11500, 480, 2.6, 0.88, 0.38],
        [12500, 510, 2.4, 0.92, 0.42],
        [11800, 490, 2.55, 0.89, 0.39],
        [12200, 505, 2.45, 0.91, 0.41]
    ]
    
    # 公交车特征 - 大面积，中等长宽比
    bus_features = [
        [15000, 550, 2.0, 0.95, 0.35],
        [14500, 540, 2.1, 0.94, 0.34],
        [15500, 560, 1.9, 0.96, 0.36],
        [14800, 545, 2.05, 0.93, 0.33],
        [15200, 555, 1.95, 0.94, 0.35]
    ]
    
    # 摩托车特征 - 小面积，大长宽比
    motorcycle_features = [
        [2000, 200, 1.8, 0.7, 0.5],
        [1800, 190, 1.9, 0.68, 0.48],
        [2200, 210, 1.7, 0.72, 0.52],
        [1900, 195, 1.85, 0.69, 0.49],
        [2100, 205, 1.75, 0.71, 0.51]
    ]
    
    # 合并特征和标签
    features = np.vstack([car_features, truck_features, bus_features, motorcycle_features])
    labels = np.array(['car'] * len(car_features) + 
                     ['truck'] * len(truck_features) + 
                     ['bus'] * len(bus_features) + 
                     ['motorcycle'] * len(motorcycle_features))
    
    return features, labels

def train_vehicle_classifier():
    """
    训练车型分类器并返回训练好的分类器
    
    返回:
        classifier: 训练好的分类器
    """
    # 生成示例训练数据
    features, labels = generate_sample_training_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    # 创建分析器实例
    analyzer = TrafficAnalyzer(video_source=0)  # 参数不重要，只需要实例化
    
    # 训练分类器
    analyzer.train_classifier(X_train, y_train)
    
    # 评估分类器
    if hasattr(analyzer.classifier, 'score'):
        score = analyzer.classifier.score(X_test, y_test)
        print(f"分类器准确率: {score:.2f}")
    
    return analyzer.classifier

def prepare_test_video(input_path, output_path, duration=60):
    """
    准备测试视频（截取指定时长）
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        duration: 截取时长（秒）
    """
    if not os.path.exists(input_path):
        print(f"错误: 输入视频 {input_path} 不存在")
        return False
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        return False
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算需要截取的帧数
    total_frames = int(fps * duration)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 截取视频
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"视频截取完成: {output_path} ({frame_count} 帧)")
    return True

def define_roi(video_path):
    """
    交互式定义感兴趣区域(ROI)
    
    参数:
        video_path: 视频路径
        
    返回:
        roi: 感兴趣区域坐标列表
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return None
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取视频帧")
        cap.release()
        return None
    
    # 创建窗口
    cv2.namedWindow('定义ROI')
    
    # 存储ROI点
    roi_points = []
    
    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            # 在图像上绘制点
            cv2.circle(roi_image, (x, y), 5, (0, 255, 0), -1)
            # 如果有多个点，绘制线
            if len(roi_points) > 1:
                cv2.line(roi_image, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            # 如果至少有3个点，绘制闭合多边形
            if len(roi_points) > 2:
                cv2.line(roi_image, roi_points[0], roi_points[-1], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('定义ROI', roi_image)
    
    # 设置鼠标回调
    roi_image = frame.copy()
    cv2.setMouseCallback('定义ROI', mouse_callback)
    
    # 显示图像并等待用户输入
    cv2.imshow('定义ROI', roi_image)
    print("点击图像定义ROI区域，完成后按'Enter'确认，按'r'重置，按'ESC'取消")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 按Enter确认
        if key == 13:
            if len(roi_points) >= 3:
                break
            else:
                print("至少需要3个点来定义ROI")
        
        # 按r重置
        elif key == ord('r'):
            roi_points = []
            roi_image = frame.copy()
            cv2.imshow('定义ROI', roi_image)
            print("ROI已重置")
        
        # 按ESC取消
        elif key == 27:
            roi_points = []
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    return roi_points if roi_points else None

if __name__ == "__main__":
    # 示例用法
    # 1. 训练车型分类器
    classifier = train_vehicle_classifier()
    
    # 2. 准备测试视频（如果需要）
    # prepare_test_video("long_traffic_video.mp4", "traffic_video.mp4", duration=60)
    
    # 3. 定义ROI（如果需要）
    # video_path = "traffic_video.mp4"
    # roi = define_roi(video_path)
    # if roi:
    #     print(f"定义的ROI: {roi}")
    #     
    #     # 使用定义的ROI创建分析器
    #     analyzer = TrafficAnalyzer(video_source=video_path, roi=roi)
    #     analyzer.classifier = classifier  # 设置训练好的分类器
    #     analyzer.run()