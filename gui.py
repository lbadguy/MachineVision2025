import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import os
import time
from PIL import Image, ImageTk
from traffic_analyzer import TrafficAnalyzer

class TrafficAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("交通流量智能分析系统")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # 设置字体
        self.font = ("SimHei", 12)
        
        # 视频源选择框架
        self.source_frame = tk.Frame(root, bg="#f0f0f0")
        self.source_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(self.source_frame, text="请选择视频源：", font=self.font, bg="#f0f0f0").pack(side="left", padx=5)
        
        # 摄像头按钮
        self.camera_btn = tk.Button(self.source_frame, text="摄像头", font=self.font, 
                                  command=self.use_camera, width=10, bg="#4CAF50", fg="white")
        self.camera_btn.pack(side="left", padx=10)
        
        # 视频文件按钮
        self.file_btn = tk.Button(self.source_frame, text="视频文件", font=self.font, 
                                command=self.use_video_file, width=10, bg="#2196F3", fg="white")
        self.file_btn.pack(side="left", padx=10)
        
        # 视图控制按钮
        self.zoom_in_btn = tk.Button(self.source_frame, text="放大", font=self.font, 
                                  command=self.zoom_in, width=6, bg="#FF9800", fg="white")
        self.zoom_in_btn.pack(side="left", padx=10)
        
        self.zoom_out_btn = tk.Button(self.source_frame, text="缩小", font=self.font, 
                                   command=self.zoom_out, width=6, bg="#FF9800", fg="white")
        self.zoom_out_btn.pack(side="left", padx=10)
        
        # 导出数据按钮
        self.export_btn = tk.Button(self.source_frame, text="导出数据", font=self.font, 
                                 command=self.export_data, width=10, bg="#9C27B0", fg="white")
        self.export_btn.pack(side="left", padx=10)
        
        # 视频显示区域
        self.video_frame = tk.Frame(root, bg="#f0f0f0", width=1000, height=600)
        self.video_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg="#f0f0f0")
        self.video_label.pack(fill="both", expand=True)
        
        # 显示欢迎界面
        self.show_welcome_screen()
        
        # 控制按钮区域
        self.control_frame = tk.Frame(root, bg="#f0f0f0", height=100)
        self.control_frame.pack(fill="x", padx=20, pady=10)
        
        # 后退5秒按钮
        self.back_btn = tk.Button(self.control_frame, text="⏪", font=("Arial", 16), 
                                command=self.back_5_seconds, width=3, state="disabled")
        self.back_btn.pack(side="left", padx=10)
        
        # 暂停/播放按钮
        self.play_pause_btn = tk.Button(self.control_frame, text="⏸", font=("Arial", 16), 
                                     command=self.toggle_play_pause, width=3, state="disabled")
        self.play_pause_btn.pack(side="left", padx=10)
        
        # 前进5秒按钮
        self.forward_btn = tk.Button(self.control_frame, text="⏩", font=("Arial", 16), 
                                   command=self.forward_5_seconds, width=3, state="disabled")
        self.forward_btn.pack(side="left", padx=10)
        
        # 交通信息显示区域
        self.info_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        self.info_frame.pack(side="left", padx=20, fill="y")
        
        self.traffic_info_label = tk.Label(self.info_frame, text="交通信息", font=("SimHei", 14, "bold"), bg="#f0f0f0")
        self.traffic_info_label.pack(anchor="w")
        
        self.vehicle_count_label = tk.Label(self.info_frame, text="车辆数量: 0", font=self.font, bg="#f0f0f0")
        self.vehicle_count_label.pack(anchor="w")
        
        self.avg_speed_label = tk.Label(self.info_frame, text="平均速度: 0.00", font=self.font, bg="#f0f0f0")
        self.avg_speed_label.pack(anchor="w")
        
        self.traffic_status_label = tk.Label(self.info_frame, text="交通状况: 畅通", font=self.font, bg="#f0f0f0")
        self.traffic_status_label.pack(anchor="w")
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(self.control_frame, orient="horizontal", length=500, 
                                    variable=self.progress_var, command=self.seek_video)
        self.progress_bar.pack(side="left", padx=20, fill="x", expand=True)
        self.progress_bar.config(state="disabled")
        
        # 时间标签
        self.time_label = tk.Label(self.control_frame, text="00:00 / 00:00", font=self.font, bg="#f0f0f0")
        self.time_label.pack(side="left", padx=10)
        
        # 初始化变量
        self.cap = None
        self.analyzer = None
        self.is_playing = False
        self.is_camera = False
        self.video_path = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame_id = 0
        self.update_id = None
        
        # 视图控制变量
        self.zoom_factor = 1.0
        self.roi = None
        self.is_selecting_roi = False
        self.roi_start_point = None
        
    def use_camera(self):
        """使用摄像头作为视频源"""
        if self.cap is not None:
            self.stop_video()
        
        # 清除欢迎界面
        for widget in self.video_label.winfo_children():
            widget.destroy()
        self.video_label.config(bg="black")
        self.video_frame.config(bg="black")
            
        self.is_camera = True
        self.video_path = 0
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            tk.messagebox.showerror("错误", "无法打开摄像头")
            return
        
        # 初始化分析器
        self.analyzer = TrafficAnalyzer(video_source=0)
        
        # 更新按钮状态
        self.back_btn.config(state="disabled")
        self.forward_btn.config(state="disabled")
        self.progress_bar.config(state="disabled")
        self.play_pause_btn.config(state="normal", text="⏸")
        
        # 开始播放
        self.is_playing = True
        self.update_frame()
        
    def use_video_file(self):
        """使用视频文件作为视频源"""
        if self.cap is not None:
            self.stop_video()
        
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")])
        if not file_path:
            return
        
        # 清除欢迎界面
        for widget in self.video_label.winfo_children():
            widget.destroy()
        self.video_label.config(bg="black")
        self.video_frame.config(bg="black")
            
        self.is_camera = False
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            tk.messagebox.showerror("错误", f"无法打开视频文件: {file_path}")
            return
        
        # 获取视频信息
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_id = 0
        
        # 初始化分析器
        self.analyzer = TrafficAnalyzer(video_source=file_path)
        
        # 更新按钮状态
        self.back_btn.config(state="normal")
        self.forward_btn.config(state="normal")
        self.progress_bar.config(state="normal")
        self.play_pause_btn.config(state="normal", text="⏸")
        
        # 更新进度条范围
        self.progress_bar.config(from_=0, to=self.total_frames-1)
        
        # 读取第一帧并显示
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_id = 1
            self.progress_var.set(0)
            self.display_frame(frame)
            self.update_time_label()
        
        # 暂停状态开始
        self.is_playing = False
        self.toggle_play_pause()
        
    def update_frame(self):
        """更新视频帧"""
        if self.cap is None or not self.is_playing:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            if self.is_camera:
                # 摄像头可能暂时无法读取，继续尝试
                self.update_id = self.root.after(30, self.update_frame)
                return
            else:
                # 视频文件播放结束
                self.is_playing = False
                self.play_pause_btn.config(text="▶")
                return
        
        # 更新当前帧ID和进度条
        if not self.is_camera:
            self.current_frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_var.set(self.current_frame_id)
            self.update_time_label()
        
        # 显示帧
        self.display_frame(frame)
        
        # 继续更新
        self.update_id = self.root.after(30, self.update_frame)
        
    def display_frame(self, frame, draw_roi=True):
        """显示视频帧"""
        try:
            if draw_roi:
                # 检测车辆
                contours = self.analyzer.detect_vehicles(frame)
                
                # 跟踪车辆
                self.analyzer.track_vehicles(contours, frame)
                
                # 分析交通状况
                traffic_info = self.analyzer.analyze_traffic()
                
                # 更新交通信息标签
                self.update_traffic_info_labels(traffic_info)
                
                # 可视化
                visualization = self.analyzer.visualize_traffic(frame, traffic_info)
                
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            else:
                # 直接转换颜色空间，不进行分析
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应显示区域
            h, w, _ = frame_rgb.shape
            video_w = self.video_frame.winfo_width()
            video_h = self.video_frame.winfo_height()
            
            # 确保窗口尺寸有效
            if video_w <= 1 or video_h <= 1:
                video_w = max(640, w)
                video_h = max(480, h)
            
            # 应用缩放因子
            if self.zoom_factor != 1.0:
                # 计算缩放后的尺寸
                zoom_w = int(w * self.zoom_factor)
                zoom_h = int(h * self.zoom_factor)
                
                # 计算裁剪区域（居中）
                start_x = max(0, (zoom_w - w) // 2)
                start_y = max(0, (zoom_h - h) // 2)
                end_x = min(zoom_w, start_x + w)
                end_y = min(zoom_h, start_y + h)
                
                # 缩放图像
                if self.zoom_factor > 1.0:
                    # 放大：先调整大小，再裁剪中心区域
                    frame_zoomed = cv2.resize(frame_rgb, (zoom_w, zoom_h))
                    frame_rgb = frame_zoomed[start_y:end_y, start_x:end_x]
                else:
                    # 缩小：直接调整大小
                    frame_rgb = cv2.resize(frame_rgb, (zoom_w, zoom_h))
            
            # 保持宽高比
            ratio = min(video_w/frame_rgb.shape[1], video_h/frame_rgb.shape[0])
            new_w = max(1, int(frame_rgb.shape[1] * ratio))
            new_h = max(1, int(frame_rgb.shape[0] * ratio))
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            # 转换为PhotoImage
            img = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # 更新标签
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
        except Exception as e:
            print(f"显示帧时出错: {e}")
            import traceback
            traceback.print_exc()
        
    def toggle_play_pause(self):
        """切换播放/暂停状态"""
        if self.cap is None:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_pause_btn.config(text="⏸")
            self.update_frame()
        else:
            self.play_pause_btn.config(text="▶")
            if self.update_id is not None:
                self.root.after_cancel(self.update_id)
                self.update_id = None
    
    def back_5_seconds(self):
        """后退5秒"""
        if self.cap is None or self.is_camera:
            return
            
        # 计算要后退的帧数
        frames_to_move = int(self.fps * 5)
        target_frame = max(0, self.current_frame_id - frames_to_move)
        
        # 设置视频位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame_id = target_frame
        self.progress_var.set(target_frame)
        
        # 读取并显示当前帧
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_time_label()
    
    def forward_5_seconds(self):
        """前进5秒"""
        if self.cap is None or self.is_camera:
            return
            
        # 计算要前进的帧数
        frames_to_move = int(self.fps * 5)
        target_frame = min(self.total_frames - 1, self.current_frame_id + frames_to_move)
        
        # 设置视频位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame_id = target_frame
        self.progress_var.set(target_frame)
        
        # 读取并显示当前帧
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_time_label()
    
    def update_traffic_info_labels(self, traffic_info):
        """更新交通信息标签"""
        # 更新车辆数量
        vehicle_count = traffic_info.get('vehicle_count', 0)
        self.vehicle_count_label.config(text=f"车辆数量: {vehicle_count}")
        
        # 更新平均速度
        avg_speed = traffic_info.get('avg_speed', 0)
        self.avg_speed_label.config(text=f"平均速度: {avg_speed:.2f}")
        
        # 更新交通状况
        congestion = traffic_info.get('congestion', False)
        self.traffic_status_label.config(
            text=f"交通状况: {'拥堵' if congestion else '畅通'}",
            fg="#FF0000" if congestion else "#008000"
        )
    
    def seek_video(self, value):
        """拖动进度条定位视频"""
        if self.cap is None or self.is_camera:
            return
            
        # 暂停视频
        was_playing = self.is_playing
        if was_playing:
            self.toggle_play_pause()
        
        # 设置视频位置
        frame_id = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.current_frame_id = frame_id
        
        # 读取并显示当前帧
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_time_label()
        
        # 如果之前是播放状态，恢复播放
        if was_playing:
            self.toggle_play_pause()
    
    def update_time_label(self):
        """更新时间标签"""
        if self.cap is None or self.is_camera:
            self.time_label.config(text="--:-- / --:--")
            return
            
        # 计算当前时间和总时间
        current_seconds = self.current_frame_id / self.fps
        total_seconds = self.total_frames / self.fps
        
        # 格式化时间
        current_time = time.strftime("%M:%S", time.gmtime(current_seconds))
        total_time = time.strftime("%M:%S", time.gmtime(total_seconds))
        
        # 更新标签
        self.time_label.config(text=f"{current_time} / {total_time}")
    
    def stop_video(self):
        """停止视频播放并释放资源"""
        if self.is_playing and self.update_id is not None:
            self.root.after_cancel(self.update_id)
            self.update_id = None
        
        self.is_playing = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def zoom_in(self):
        """放大视图"""
        self.zoom_factor *= 1.2
        if self.cap is not None and not self.is_playing:
            # 如果视频已暂停，更新当前帧显示
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id - 1)
                self.display_frame(frame)
    
    def zoom_out(self):
        """缩小视图"""
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.5:
            self.zoom_factor = 0.5
        if self.cap is not None and not self.is_playing:
            # 如果视频已暂停，更新当前帧显示
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id - 1)
                self.display_frame(frame)
    
    # 框选区域功能已移除
    
    def show_welcome_screen(self):
        """显示欢迎界面"""
        # 创建欢迎界面画布
        welcome_canvas = tk.Canvas(self.video_label, bg="#f0f0f0", highlightthickness=0)
        welcome_canvas.pack(fill="both", expand=True)
        
        # 系统标题
        welcome_canvas.create_text(
            500, 200, 
            text="欢迎来到 交通流量智能分析系统", 
            font=("SimHei", 36, "bold"), 
            fill="#2196F3"
        )
        
        # 系统简介
        welcome_canvas.create_text(
            500, 280, 
            text="项目基于计算机视觉技术的智能交通监控与分析平台", 
            font=("SimHei", 18), 
            fill="#333333"
        )
        
        # 使用提示
        welcome_canvas.create_text(
            500, 350, 
            text="请选择上方按钮以开始使用系统：", 
            font=("SimHei", 16), 
            fill="#555555"
        )
        
        # 功能说明
        welcome_canvas.create_text(
            500, 400, 
            text="【摄像头】- 使用计算机摄像头进行实时交通分析\n【视频文件】- 加载本地视频文件进行交通分析", 
            font=("SimHei", 14), 
            fill="#555555",
            justify="center"
        )
        
        # 我们的信息
        welcome_canvas.create_text(
            500, 550, 
            text="2025 机器视觉课程设计", 
            font=("SimHei", 10), 
            fill="#999999"
        )

        welcome_canvas.create_text(
            500, 570, 
            text="22计科2班 梁桂诚", 
            font=("SimHei", 10), 
            fill="#999999"
        )
    
    def export_data(self):
        """导出交通数据到CSV文件"""
        if self.analyzer is None:
            tk.messagebox.showinfo("提示", "请先选择视频源并开始分析")
            return
            
        # 调用分析器的导出方法
        export_path = self.analyzer.export_all_data()
        
        if export_path is None:
            tk.messagebox.showinfo("提示", "没有找到可导出的交通数据")
        else:
            # 显示导出成功消息
            tk.messagebox.showinfo("导出成功", f"交通数据已成功导出到:\n{export_path}")
            
            # 询问是否打开文件所在目录
            if tk.messagebox.askyesno("打开文件夹", "是否打开导出文件所在目录?"):
                import subprocess
                # 使用系统默认方式打开文件夹
                subprocess.Popen(f'explorer "{os.path.dirname(export_path)}"')
    
    def on_closing(self):
        """窗口关闭时的处理"""
        self.stop_video()
        self.root.destroy()

def main():
    # 设置中文显示
    import locale
    import platform
    
    # 根据操作系统设置适当的编码
    system = platform.system()
    try:
        if system == 'Windows':
            # Windows系统设置中文编码
            locale.setlocale(locale.LC_ALL, 'Chinese_China')
        elif system == 'Linux' or system == 'Darwin':  # Darwin是macOS
            # Linux/macOS系统设置中文编码
            try:
                locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
            except locale.Error:
                locale.setlocale(locale.LC_ALL, 'zh_CN.utf8')
    except Exception as e:
        print(f"警告: 无法设置中文编码环境: {e}")
        print("程序将继续运行，但可能会出现中文显示异常")
    
    try:
        root = tk.Tk()
        app = TrafficAnalyzerGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()