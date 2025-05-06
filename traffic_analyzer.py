import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
# Import configuration variables
from config import (BG_HISTORY, BG_THRESHOLD, DETECT_SHADOWS, MIN_CONTOUR_AREA, 
                   MAX_CONTOUR_AREA, MORPH_KERNEL_TYPE, MORPH_KERNEL_SIZE, 
                   DILATE_ITERATIONS, CLOSE_ITERATIONS, RECORD_INTERVAL, 
                   CONGESTION_VEHICLE_COUNT, CONGESTION_SPEED_THRESHOLD, 
                   DATA_DIR, VEHICLE_COLORS, SHOW_BOUNDING_BOXES, 
                   SHOW_VEHICLE_ID, SHOW_VEHICLE_SPEED)

class TrafficAnalyzer:
    def __init__(self, video_source=0, roi=None, save_data=True):
        """
        初始化交通分析器
        
        参数:
            video_source: 视频源（可以是摄像头索引或视频文件路径）
            roi: 感兴趣区域，格式为[(x1,y1), (x2,y2), ...]
            save_data: 是否保存交通数据用于后续分析
        """
        self.video_source = video_source
        self.roi = roi
        self.save_data = save_data
        
        # 初始化背景减除器（使用配置文件中的参数）
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY, 
            varThreshold=BG_THRESHOLD, 
            detectShadows=DETECT_SHADOWS
        )
        
        # 车辆跟踪相关变量
        self.vehicles = []
        self.vehicle_centers = []  # 存储车辆中心点用于计数
        self.next_vehicle_id = 0
        self.min_contour_area = MIN_CONTOUR_AREA  # 最小车辆轮廓面积
        self.max_contour_area = MAX_CONTOUR_AREA  # 最大车辆轮廓面积
        
        # 车辆分类器
        self.classifier = None
        
        # 交通数据记录
        self.traffic_data = []
        self.last_record_time = time.time()
        self.record_interval = RECORD_INTERVAL  # 记录间隔（秒）
        
        # 拥堵判断阈值
        self.congestion_vehicle_count = CONGESTION_VEHICLE_COUNT  # 车辆数阈值
        self.congestion_speed_threshold = CONGESTION_SPEED_THRESHOLD  # 速度阈值（像素/帧）
        
        # 创建数据保存目录
        if self.save_data:
            os.makedirs(DATA_DIR, exist_ok=True)
    
    def train_classifier(self, features, labels):
        """
        训练车型分类器
        
        参数:
            features: 特征矩阵，每行代表一个车辆的特征
            labels: 标签向量，对应每个车辆的类型
        """
        self.classifier = DecisionTreeClassifier(max_depth=5)
        self.classifier.fit(features, labels)
        print("分类器训练完成")
    
    def detect_vehicles(self, frame):
        """
        使用背景减除法检测车辆
        
        参数:
            frame: 当前视频帧
            
        返回:
            contours: 检测到的车辆轮廓
        """
        # 应用ROI掩码（如果有）
        if self.roi is not None:
            mask = np.zeros_like(frame[:, :, 0])
            cv2.fillPoly(mask, [np.array(self.roi, dtype=np.int32)], 255)
            frame_roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            frame_roi = frame
        
        # 灰度处理
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        
        # 高斯去噪（参考互联网资源中的预处理方法）
        blur = cv2.GaussianBlur(frame_roi, (3, 3), 5)
        
        # 应用背景减除
        fg_mask = self.bg_subtractor.apply(blur)
        
        # 去除阴影（MOG2检测的阴影值为127）
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪（使用配置文件中的参数）
        kernel_type = cv2.MORPH_RECT if MORPH_KERNEL_TYPE == 0 else cv2.MORPH_CROSS if MORPH_KERNEL_TYPE == 1 else cv2.MORPH_ELLIPSE
        kernel = cv2.getStructuringElement(kernel_type, MORPH_KERNEL_SIZE)
        
        # 先腐蚀再膨胀（开操作，去除小噪点）
        fg_mask = cv2.erode(fg_mask, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=DILATE_ITERATIONS)
        
        # 连续多次闭操作（填充目标内部空洞）
        for _ in range(CLOSE_ITERATIONS):
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓（基于面积）
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def track_vehicles(self, contours, frame):
        """
        跟踪检测到的车辆
        
        参数:
            contours: 检测到的车辆轮廓
            frame: 当前视频帧
        """
        # 当前帧检测到的车辆
        current_vehicles = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 提取车辆特征
            features = self.extract_features(contour, w, h)
            
            # 尝试匹配已有车辆
            matched = False
            for vehicle in self.vehicles:
                # 计算中心点距离
                dist = np.sqrt((center_x - vehicle['center_x'])**2 + (center_y - vehicle['center_y'])**2)
                
                # 如果距离小于阈值，认为是同一辆车
                if dist < 50:  # 距离阈值
                    # 更新车辆信息
                    prev_x, prev_y = vehicle['center_x'], vehicle['center_y']
                    vehicle['center_x'] = center_x
                    vehicle['center_y'] = center_y
                    vehicle['contour'] = contour
                    vehicle['bbox'] = (x, y, w, h)
                    
                    # 计算速度（像素/帧）
                    dx = center_x - prev_x
                    dy = center_y - prev_y
                    vehicle['speed'] = np.sqrt(dx**2 + dy**2)
                    
                    # 更新特征
                    vehicle['features'] = features
                    
                    # 如果有分类器，进行车型分类
                    if self.classifier is not None:
                        vehicle['type'] = self.classifier.predict([features])[0]
                    
                    # 更新跟踪时间
                    vehicle['last_seen'] = time.time()
                    
                    current_vehicles.append(vehicle)
                    matched = True
                    break
            
            # 如果没有匹配到已有车辆，创建新车辆
            if not matched:
                vehicle = {
                    'id': self.next_vehicle_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'speed': 0,
                    'features': features,
                    'type': 'unknown',
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
                
                # 如果有分类器，进行车型分类
                if self.classifier is not None:
                    vehicle['type'] = self.classifier.predict([features])[0]
                
                current_vehicles.append(vehicle)
                self.next_vehicle_id += 1
                
            # 车辆跟踪逻辑结束
        
        # 更新车辆列表，移除长时间未检测到的车辆
        self.vehicles = [v for v in current_vehicles if time.time() - v['last_seen'] < 1.0]
    
    def extract_features(self, contour, width, height):
        """
        提取车辆特征用于分类
        
        参数:
            contour: 车辆轮廓
            width: 边界框宽度
            height: 边界框高度
            
        返回:
            features: 特征向量
        """
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 计算周长
        perimeter = cv2.arcLength(contour, True)
        
        # 计算长宽比
        aspect_ratio = float(width) / height if height > 0 else 0
        
        # 计算矩形度（轮廓面积与边界矩形面积之比）
        rect_area = width * height
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # 计算圆形度
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 返回特征向量
        return [area, perimeter, aspect_ratio, extent, circularity]
    
    def analyze_traffic(self):
        """
        分析当前交通状况
        
        返回:
            traffic_info: 包含车辆数、平均速度、交通状况等信息的字典
        """
        if not self.vehicles:
            return {
                'vehicle_count': 0,
                'avg_speed': 0,
                'congestion': False
            }
        
        # 计算车辆数
        vehicle_count = len(self.vehicles)
        
        # 计算平均速度
        speeds = [v['speed'] for v in self.vehicles]
        avg_speed = np.mean(speeds) if speeds else 0
        
        # 判断是否拥堵
        is_congested = (vehicle_count >= self.congestion_vehicle_count and 
                        avg_speed <= self.congestion_speed_threshold)
        
        # 不再计算通过检测线的车辆数
        
        # 记录交通数据
        if self.save_data and time.time() - self.last_record_time >= self.record_interval:
            self.record_traffic_data(vehicle_count, avg_speed, {}, is_congested)
            self.last_record_time = time.time()
        
        return {
            'vehicle_count': vehicle_count,
            'avg_speed': avg_speed,
            'congestion': is_congested
        }
    
    def record_traffic_data(self, vehicle_count, avg_speed, vehicle_types, is_congested):
        """
        记录交通数据
        
        参数:
            vehicle_count: 车辆数
            avg_speed: 平均速度
            vehicle_types: 车型分布（已弃用，保留参数兼容性）
            is_congested: 是否拥堵
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            'timestamp': timestamp,
            'vehicle_count': vehicle_count,
            'avg_speed': avg_speed,
            'is_congested': is_congested
        }
        self.traffic_data.append(data)
        
        # 保存到文件
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"traffic_data/traffic_{date_str}.csv"
        
        # 如果文件不存在，创建并写入表头
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("timestamp,vehicle_count,avg_speed,is_congested\n")
        
        # 写入数据
        with open(filename, 'a') as f:
            f.write(f"{timestamp},{vehicle_count},{avg_speed},{is_congested}\n")
            
    def export_all_data(self, export_path=None):
        """
        导出所有交通数据到CSV文件
        
        参数:
            export_path: 导出文件路径，如果为None则使用默认路径
            
        返回:
            导出文件的路径
        """
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
            
        if export_path is None:
            # 使用当前时间作为文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"{DATA_DIR}/traffic_export_{timestamp}.csv"
        
        # 获取所有CSV文件
        csv_files = []
        for file in os.listdir(DATA_DIR):
            if file.startswith("traffic_") and file.endswith(".csv"):
                csv_files.append(os.path.join(DATA_DIR, file))
        
        # 如果没有数据文件，返回None
        if not csv_files:
            return None
            
        # 合并所有数据
        all_data = []
        header = "timestamp,vehicle_count,avg_speed,is_congested"
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    # 跳过表头
                    if lines and lines[0].strip() == header:
                        all_data.extend(lines[1:])
                    else:
                        all_data.extend(lines)
            except Exception as e:
                print(f"读取文件 {csv_file} 时出错: {e}")
        
        # 写入合并后的数据
        try:
            with open(export_path, 'w') as f:
                # 写入表头
                f.write(header + "\n")
                # 写入数据
                for line in all_data:
                    f.write(line)
            return export_path
        except Exception as e:
            print(f"导出数据时出错: {e}")
            return None
    
    def predict_peak_hours(self):
        """
        基于历史数据预测交通高峰时段
        
        返回:
            peak_hours: 预测的高峰时段列表
        """
        if not self.traffic_data:
            return []
        
        # 按小时统计车流量
        hourly_counts = defaultdict(list)
        for data in self.traffic_data:
            timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S")
            hour = timestamp.hour
            hourly_counts[hour].append(data['vehicle_count'])
        
        # 计算每小时平均车流量
        hourly_avg_counts = {}
        for hour, counts in hourly_counts.items():
            hourly_avg_counts[hour] = np.mean(counts)
        
        # 找出车流量最大的几个小时
        sorted_hours = sorted(hourly_avg_counts.items(), key=lambda x: x[1], reverse=True)
        peak_hours = [hour for hour, _ in sorted_hours[:3]]  # 取前三个高峰时段
        
        return peak_hours
    
    def visualize_traffic(self, frame, traffic_info):
        """
        可视化交通状况
        
        参数:
            frame: 当前视频帧
            traffic_info: 交通分析信息
            
        返回:
            visualization: 可视化后的帧
        """
        # 创建可视化图像
        visualization = frame.copy()
        
        # 绘制ROI区域（如果有）
        if self.roi is not None:
            cv2.polylines(visualization, [np.array(self.roi, dtype=np.int32)], True, (0, 255, 0), 2)
        
        # 绘制检测到的车辆
        for vehicle in self.vehicles:
            x, y, w, h = vehicle['bbox']
            speed = vehicle['speed']
            center_x, center_y = vehicle['center_x'], vehicle['center_y']
            
            # 根据车型选择颜色
            if SHOW_BOUNDING_BOXES:
                vehicle_type = vehicle.get('type', 'unknown')
                color = VEHICLE_COLORS.get(vehicle_type, (0, 255, 0))
                
                # 绘制边界框
                cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
                
                # 绘制中心点
                cv2.circle(visualization, (center_x, center_y), 4, color, -1)
                
                # 绘制车辆ID
                if SHOW_VEHICLE_ID:
                    id_label = f"ID: {vehicle['id']}"
                    self.put_chinese_text(visualization, id_label, (x, y - 10), color, 20)
                
                # 绘制速度
                if SHOW_VEHICLE_SPEED:
                    speed_label = f"速度: {speed:.1f}"
                    self.put_chinese_text(visualization, speed_label, (x, y + h + 20), color, 20)
        
        # 创建热力图覆盖层
        if len(self.vehicles) > 0:
            # 创建热力图
            heatmap = np.zeros_like(frame[:, :, 0]).astype(np.float32)
            
            # 为每个车辆位置添加高斯分布
            for vehicle in self.vehicles:
                x, y, w, h = vehicle['bbox']
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(heatmap, (center_x, center_y), max(w, h) // 2, 1, -1)
            
            # 应用高斯模糊
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            
            # 归一化热力图
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = heatmap.astype(np.uint8)
            
            # 应用颜色映射
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 添加热力图覆盖层
            alpha = 0.3
            visualization = cv2.addWeighted(visualization, 1 - alpha, heatmap, alpha, 0)
        
        # 绘制交通信息
        vehicle_count = traffic_info.get('vehicle_count', 0)
        avg_speed = traffic_info.get('avg_speed', 0)
        congestion = traffic_info.get('congestion', False)
        
        info_text = [
            f"当前车辆数: {vehicle_count}",
            f"平均速度: {avg_speed:.2f}",
            f"交通状况: {'拥堵' if congestion else '畅通'}"
        ]
        
        # 绘制信息文本
        for i, text in enumerate(info_text):
            self.put_chinese_text(visualization, text, (10, 30 + i * 30), (255, 255, 255), 24)
        
        # 如果检测到拥堵，在右下角显示警告（带半透明背景）
        if congestion:
            warning_text = "警告: 交通拥堵!"
            # 计算文本大小以确定背景区域
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            # 计算右下角位置
            text_x = visualization.shape[1] - text_size[0] - 20
            text_y = visualization.shape[0] - 30
            
            # 绘制半透明背景
            overlay = visualization.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            # 应用透明度
            alpha = 0.6
            visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)
            
            # 绘制警告文本 - 调整文字位置使其居中显示在背景中
            self.put_chinese_text(visualization, warning_text, (text_x, text_y - text_size[1]//2), (0, 0, 255), 36)
        
        return visualization
        
    def put_chinese_text(self, img, text, position, color, size):
        """
        在图像上绘制中文文本
        
        参数:
            img: 图像
            text: 文本
            position: 位置 (x, y)
            color: 颜色 (B, G, R)
            size: 字体大小
        """
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)
        
        # 加载中文字体
        try:
            # 尝试使用系统中的中文字体
            if os.path.exists('C:/Windows/Fonts/simhei.ttf'):
                font = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', size)
            elif os.path.exists('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
                font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size)
            else:
                # 如果找不到中文字体，使用默认字体
                font = ImageFont.load_default()
                print("警告: 找不到中文字体，使用默认字体")
        except Exception as e:
            print(f"加载字体出错: {e}")
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        
        # 将PIL图像转换回OpenCV图像
        img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        
    def get_frame(self, frame_id=None):
        """
        获取指定帧并进行分析
        
        参数:
            frame_id: 帧ID，如果为None则获取当前帧
            
        返回:
            frame: 分析后的帧
            traffic_info: 交通信息
        """
        # 创建临时视频捕获对象
        if isinstance(self.video_source, int):
            # 如果是摄像头，无法跳转到指定帧
            return None, None
            
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            return None, None
            
        # 如果指定了帧ID，跳转到该帧
        if frame_id is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
            
        # 检测车辆
        contours = self.detect_vehicles(frame)
        
        # 跟踪车辆
        self.track_vehicles(contours, frame)
        
        # 分析交通状况
        traffic_info = self.analyze_traffic()
        
        # 可视化
        visualization = self.visualize_traffic(frame, traffic_info)
        
        # 释放资源
        cap.release()
        
        return visualization, traffic_info
        

        
    def get_frame(self, frame_id=None):
        """
        获取指定帧并进行分析
        
        参数:
            frame_id: 帧ID，如果为None则获取当前帧
            
        返回:
            frame: 分析后的帧
            traffic_info: 交通信息
        """
        # 创建临时视频捕获对象
        if isinstance(self.video_source, int):
            # 如果是摄像头，无法跳转到指定帧
            return None, None
            
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            return None, None
            
        # 如果指定了帧ID，跳转到该帧
        if frame_id is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
            
        # 检测车辆
        contours = self.detect_vehicles(frame)
        
        # 跟踪车辆
        self.track_vehicles(contours, frame)
        
        # 分析交通状况
        traffic_info = self.analyze_traffic()
        
        # 可视化
        visualization = self.visualize_traffic(frame, traffic_info)
        
        # 释放资源
        cap.release()
        
        return visualization, traffic_info
    
    def run(self):
        """
        运行交通分析器
        """
        # 打开视频源
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频源 {self.video_source}")
            return
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            if not ret:
                print("视频结束或读取错误")
                break
            
            # 检测车辆
            contours = self.detect_vehicles(frame)
            
            # 跟踪车辆
            self.track_vehicles(contours, frame)
            
            # 分析交通状况
            traffic_info = self.analyze_traffic()
            
            # 可视化
            visualization = self.visualize_traffic(frame, traffic_info)
            
            # 显示结果
            cv2.imshow('交通流量智能分析', visualization)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 示例用法
    # 1. 使用摄像头
    # analyzer = TrafficAnalyzer(video_source=0)
    
    # 2. 使用视频文件
    video_path = "traffic_video.mp4"  # 替换为实际视频路径
    
    # 定义感兴趣区域（可选）
    # roi = [(100, 200), (300, 200), (500, 400), (50, 400)]
    
    analyzer = TrafficAnalyzer(video_source=video_path)
    analyzer.run()