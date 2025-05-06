import cv2
import numpy as np
import argparse
import os
import sys
import tkinter as tk
from traffic_analyzer import TrafficAnalyzer
from utils import train_vehicle_classifier, define_roi, prepare_test_video
from gui import TrafficAnalyzerGUI

# 设置字符编码
import locale
import platform

# 根据操作系统设置适当的编码
def set_locale():
    system = platform.system()
    if system == 'Windows':
        # Windows系统设置中文编码
        locale.setlocale(locale.LC_ALL, 'Chinese_China')
    elif system == 'Linux' or system == 'Darwin':  # Darwin是macOS
        # Linux/macOS系统设置中文编码
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, 'zh_CN.utf8')
            except locale.Error:
                print("警告: 无法设置中文编码环境，可能会导致中文显示异常")

def main():
    # 设置中文编码环境
    set_locale()
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='交通流量智能分析系统')
    parser.add_argument('--video', type=str, default='0', help='视频源路径或摄像头索引（默认为0，表示默认摄像头）')
    parser.add_argument('--define-roi', action='store_true', help='定义感兴趣区域')
    parser.add_argument('--train-classifier', action='store_true', help='训练车型分类器')
    parser.add_argument('--prepare-video', type=str, help='准备测试视频（截取指定视频的前60秒）')
    parser.add_argument('--output-video', type=str, help='输出视频路径（与--prepare-video一起使用）')
    parser.add_argument('--duration', type=int, default=60, help='截取视频的时长（秒）')
    parser.add_argument('--gui', action='store_true', help='启动图形用户界面')
    
    args = parser.parse_args()
    
    # 准备测试视频
    if args.prepare_video:
        output_path = args.output_video if args.output_video else 'traffic_video.mp4'
        prepare_test_video(args.prepare_video, output_path, args.duration)
        return
    
    # 启动GUI界面
    if args.gui or len(sys.argv) == 1:  # 如果指定了--gui参数或没有任何参数，启动GUI
        root = tk.Tk()
        app = TrafficAnalyzerGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        return
    
    # 命令行模式
    # 确定视频源
    if args.video.isdigit():
        video_source = int(args.video)
    else:
        video_source = args.video
        if not os.path.exists(video_source):
            print(f"错误: 视频文件 {video_source} 不存在")
            return
    
    # 定义ROI
    roi = None
    if args.define_roi:
        roi = define_roi(video_source)
        if not roi:
            print("未定义ROI，将使用整个视频帧")
    
    # 训练分类器
    classifier = None
    if args.train_classifier:
        classifier = train_vehicle_classifier()
    
    # 创建交通分析器
    analyzer = TrafficAnalyzer(video_source=video_source, roi=roi)
    
    # 设置分类器（如果有）
    if classifier:
        analyzer.classifier = classifier
    
    # 运行分析器
    analyzer.run()

if __name__ == "__main__":
    main()