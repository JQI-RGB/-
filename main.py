import sys
import random
import math
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QStatusBar, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygonF, QFont
from PyQt5.QtCore import Qt, QPointF

polygons = []#存储凸包
polygon = []#存储多边形所有顶点（也就是顶点的集合）
point_to_hull = {}#存储每个点属于哪个凸包 
#计算并返回直角三角形的斜边长度，也就是 sqrt(dx*dx + dy*dy) ，即欧氏距离。
def calculate_distance(v1, v2):
    return math.hypot(v2.x() - v1.x(), v2.y() - v1.y())

def is_colinear_and_overlap(p1, p2, q1, q2):
    # 内部函数：判断三点是否共线
    def is_colinear(a, b, c):
        # 通过叉积判断三点是否共线，允许一定的浮点误差
    
        return abs((b.y() - a.y()) * (c.x() - b.x()) - (b.x() - a.x()) * (c.y() - b.y())) < 1e-7
    # 如果q1和q2都与p1、p2共线
    if is_colinear(p1, p2, q1) and is_colinear(p1, p2, q2):
        # 判断两条线段在x和y方向上投影是否重叠（即是否有重叠部分）
        if max(min(p1.x(), p2.x()), min(q1.x(), q2.x())) <= min(max(p1.x(), p2.x()), max(q1.x(), q2.x())) + 1e-7 and \
           max(min(p1.y(), p2.y()), min(q1.y(), q2.y())) <= min(max(p1.y(), p2.y()), max(q1.y(), q2.y())) + 1e-7:
            return True  # 共线且有重叠
    return False  # 否则不共线或不重叠
#判断两个点是否相同
def is_same_point(a, b):
    return abs(a.x() - b.x()) < 1e-7 and abs(a.y() - b.y()) < 1e-7
#判断线段是否相交（不包括重合）
def segment_intersection_point(p1, p2, q1, q2):
    dx1, dy1 = p2.x() - p1.x(), p2.y() - p1.y()  # 线段p1-p2的向量
    dx2, dy2 = q2.x() - q1.x(), q2.y() - q1.y()  # 线段q1-q2的向量
    det = dx1 * dy2 - dy1 * dx2  # 行列式，判断两线段是否平行（或重合）
    if abs(det) < 1e-12:
        return None  # 行列式为0，说明平行或重合，无交点
    # 计算参数t和u，表示交点在各自线段上的比例位置
    t = ((q1.x() - p1.x()) * dy2 - (q1.y() - p1.y()) * dx2) / det
    u = ((q1.x() - p1.x()) * dy1 - (q1.y() - p1.y()) * dx1) / det
    if 0 <= t <= 1 and 0 <= u <= 1:
        # t和u都在[0,1]区间内，说明交点在两条线段的范围内
        return QPointF(p1.x() + t * dx1, p1.y() + t * dy1)  # 返回交点坐标
    return None  # 否则两线段不真正相交

def orientation(p, q, r):
    val = (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y())  # 计算三点的差积
    if abs(val) < 1e-8:
        return 0  # 共线
    return 1 if val > 0 else 2  # 1: 顺时针，2: 逆时针
#- p1 和 q1 组成第一条线段
#- p2 和 q2 组成第二条线段
def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)  # p1-q1-p2 的方向
    o2 = orientation(p1, q1, q2)  # p1-q1-q2 的方向
    o3 = orientation(p2, q2, p1)  # p2-q2-p1 的方向
    o4 = orientation(p2, q2, q1)  # p2-q2-q1 的方向
    if o1 != o2 and o3 != o4:
        return True  # 一般情况，两线段相交
    if o1 == 0 and on_segment(p1, p2, q1): return True  # 特殊情况1：p2 在 p1-q1 上
    if o2 == 0 and on_segment(p1, q2, q1): return True  # 特殊情况2：q2 在 p1-q1 上
    if o3 == 0 and on_segment(p2, p1, q2): return True  # 特殊情况3：p1 在 p2-q2 上
    if o4 == 0 and on_segment(p2, q1, q2): return True  # 特殊情况4：q1 在 p2-q2 上
    return False  # 不相交

def on_segment(p, q, r):
    # 判断点q是否在线段pr上（包含端点），允许一定的浮点误差
    return (min(p.x(), r.x()) - 1e-8 <= q.x() <= max(p.x(), r.x()) + 1e-8 and
            min(p.y(), r.y()) - 1e-8 <= q.y() <= max(p.y(), r.y()) + 1e-8 and
            abs((r.x()-p.x())*(q.y()-p.y()) - (q.x()-p.x())*(r.y()-p.y())) < 1e-8)

def crosses_any_polygon(p1, p2, polygons):
    for polygon in polygons:
        n = len(polygon)
        for i in range(n):
            q1 = polygon[i]  # 多边形的一个顶点
            q2 = polygon[(i + 1) % n]  # 下一个顶点（首尾相连）
            if is_colinear_and_overlap(p1, p2, q1, q2):
                continue  # 共线且重叠不算穿越
            if do_intersect(p1, p2, q1, q2):
                inter = segment_intersection_point(p1, p2, q1, q2)  # 求交点
                if inter is None:
                    continue  # 没有交点
                # 如果交点是端点，忽略
                if is_same_point(inter, p1) or is_same_point(inter, p2) or is_same_point(inter, q1) or is_same_point(inter, q2):
                    continue
                return True  # 存在穿越
    return False  # 没有穿越

def convex_hull(points):
    """
    计算二维点集的凸包（Graham扫描法变体）。
    输入:
        points: 点的列表，每个点为 QPointF 类型。
    返回:
        hull: 构成凸包的点的有序列表（逆时针方向）。
    """
    n = len(points)
    if n < 3:
        return []  # 少于3个点无法构成凸包
    unique_points = []
    unique_set = set()
    for point in points:
        pt = (point.x(), point.y())
        if pt not in unique_set:
            unique_set.add(pt)
            unique_points.append(point)  # 去重，避免重复点影响凸包
    n = len(unique_points)
    if n < 3:
        return []  # 去重后不足3点
    min_idx = 0
    for i in range(1, n):
        # 找到y最小的点，若y相同则x最小
        if (unique_points[i].y() < unique_points[min_idx].y() or
                (unique_points[i].y() == unique_points[min_idx].y() and
                 unique_points[i].x() < unique_points[min_idx].x())):
            min_idx = i
    sorted_points = unique_points.copy()
    # 将最小点放到首位
    sorted_points[0], sorted_points[min_idx] = sorted_points[min_idx], sorted_points[0]
    def compare_polar_angle(p):
        # 按极角排序（相对于起始点）math.atan2(y, x) 返回点 (x, y) 与 x 轴正方向的夹角（弧度）
        return math.atan2(p.y() - sorted_points[0].y(), p.x() - sorted_points[0].x())
    sorted_points[1:] = sorted(sorted_points[1:], key=compare_polar_angle)
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            p1 = hull[-2]
            p2 = hull[-1]
            p3 = sorted_points[i]
            # 判断是否为左转（逆时针），不是则弹出
            if (p2.x() - p1.x()) * (p3.y() - p1.y()) - (p2.y() - p1.y()) * (p3.x() - p1.x()) <= 0:
                hull.pop()
            else:
                break
        hull.append(sorted_points[i])  # 加入当前点
    return hull
def interpolate_bezier_closed(points, segments):
    interpolated_points = []
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        p2 = points[(i + 2) % len(points)]
        p3 = points[(i + 3) % len(points)]
        for j in range(segments + 1):
            t = j / segments
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                    (2 * p1.x()) +
                    (-p0.x() + p2.x()) * t +
                    (2 * p0.x() - 5 * p1.x() + 4 * p2.x() - p3.x()) * t2 +
                    (-p0.x() + 3 * p1.x() - 3 * p2.x() + p3.x()) * t3
            )
            y = 0.5 * (
                    (2 * p1.y()) +
                    (-p0.y() + p2.y()) * t +
                    (2 * p0.y() - 5 * p1.y() + 4 * p2.y() - p3.y()) * t2 +
                    (-p0.y() + 3 * p1.y() - 3 * p2.y() + p3.y()) * t3
            )
            interpolated_points.append(QPointF(x, y))
    return interpolated_points
#distance_matrix ：存储所有点对的最短距离
#next_matrix ：存储所有点对的最短路径上的下一个点
def update_distance_matrix(distance_matrix, next_matrix):
    n = len(distance_matrix)
    for k in range(n):  # 以每个点为中转点
        for i in range(n):  # 起点
            for j in range(n):  # 终点
                # 如果 i 到 k、k 到 j 都可达，且 i 到 j 的距离可以通过 k 得到更短路径
                if (distance_matrix[i][k] != float('inf') and
                        distance_matrix[k][j] != float('inf') and
                        distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]):
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]  # 更新最短距离
                    next_matrix[i][j] = next_matrix[i][k]  # 路径中 i 到 j 的下一个点设为 i 到 k 的下一个点

def construct_path(u, v, points, next_matrix):
    if next_matrix[u][v] == -1:  # 无法到达
        return []
    path = [points[u]]  # 路径起点
    safety = 0  # 防止死循环
    while u != v:
        u = next_matrix[u][v]  # 下一个点
        path.append(points[u])  # 加入路径
        safety += 1
        if safety > len(points) * 2:  # 安全保护，防止异常死循环
            break
    return path
#points ：点的列表
#polygons ：凸包的列表（）
#point_to_hull ：每个点属于哪个凸包的字典
#start_end_indices ：起点和终点在点列表中的索引
def compute_distance_matrix(points, polygons, point_to_hull, start_end_indices):
    n = len(points)
    distance_matrix = [[float('inf') for _ in range(n)] for _ in range(n)]
    # 1. 同一凸包只允许相邻顶点连边
    for hull_idx, poly in enumerate(polygons):  # hull_idx: 当前凸包的编号，poly: 当前凸包的顶点列表
        m = len(poly)
        for k in range(m):
            a = poly[k]  # 当前凸包的第k个顶点
            b = poly[(k + 1) % m]  # 当前凸包的下一个顶点（首尾相连）
            idx_a = points.index(a)  # 顶点a在所有点列表中的索引
            idx_b = points.index(b)  # 顶点b在所有点列表中的索引
            distance_matrix[idx_a][idx_b] = calculate_distance(a, b)  # 相邻顶点之间连边，赋值距离
            distance_matrix[idx_b][idx_a] = calculate_distance(a, b)  # 无向图，双向赋值
    # 2. 起点终点与所有点、不同凸包之间只要不穿越障碍物可连边
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
                continue
            # 如果i,j属于同一个凸包且不是相邻点，则不能连边
            if i in point_to_hull and j in point_to_hull and point_to_hull[i] == point_to_hull[j]:
                poly = polygons[point_to_hull[i]]
                idx_i = poly.index(points[i])
                idx_j = poly.index(points[j])
                if abs(idx_i - idx_j) == 1 or abs(idx_i - idx_j) == len(poly) - 1:
                    continue  # 已在上面添加
                else:
                    continue  # 非相邻
            # 起点/终点与其他点，或不同凸包之间，只要不穿越障碍物可连边
            # 防止起点终点与障碍物内部顶点连线：必须保证这两个点都不属于同一个凸包
            if (i in point_to_hull and j in point_to_hull and point_to_hull[i] == point_to_hull[j]):
                continue
            if crosses_any_polygon(points[i], points[j], polygons):
                continue
            distance_matrix[i][j] = calculate_distance(points[i], points[j])
    return distance_matrix

class PathPlanningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("路径规划应用")
        self.setGeometry(100, 100, 1700, 1000)
        self.toolbar = self.addToolBar("工具栏")
        self.toolbar.setMovable(False)
        self.reset_button = QPushButton("重置")
        self.reset_button.setMinimumHeight(35)
        self.reset_button.setMinimumWidth(120)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.undo_button = QPushButton("撤销")
        self.undo_button.setMinimumHeight(35)
        self.undo_button.setMinimumWidth(120)
        self.undo_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.toolbar.addWidget(self.reset_button)
        self.toolbar.addWidget(self.undo_button)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.drawing_area = QWidget()
        self.drawing_area.setMinimumSize(1700, 1000)
        self.main_layout.addWidget(self.drawing_area)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("请点击设置起点")
        self.reset_button.clicked.connect(self.reset)
        self.undo_button.clicked.connect(self.undo)
        self.start = None
        self.end = None
        self.start_set = False
        self.end_set = False
        self.shapes_drawn = False # 障碍物是否已绘制
        self.path = []
        self.points_history = []#用户点击历史
        global polygons, polygon, point_to_hull
        polygons = []
        polygon = []
        point_to_hull = {}

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipRect(self.drawing_area.geometry())
        # 画障碍物
        if not self.shapes_drawn:
            self.draw_shapes(painter, 30, 100, 30, 200)
            self.draw_shapes(painter, 145, 300, 100, 200)
            self.draw_shapes(painter, 100, 350, 250, 350)
            self.draw_shapes(painter, 400, 500, 200, 300)
            self.draw_shapes(painter, 500, 700, 300, 400)
            self.draw_shapes(painter, 550, 875, 30, 275)
            self.draw_shapes(painter, 750, 875, 550, 700)
            self.draw_shapes(painter, 700, 900, 400, 500)
            self.draw_shapes(painter, 900, 1000, 200, 400)
            self.draw_shapes(painter, 1100, 1500, 200, 500)
            self.draw_shapes(painter, 1400, 1700, 500, 700)
            self.draw_shapes(painter, 1400, 1700, 800, 900)
            self.draw_shapes(painter, 25, 235, 450, 700)
            self.draw_shapes(painter, 275, 500, 400, 470)
            self.draw_shapes(painter, 100, 225, 800, 900)
            self.draw_shapes(painter, 270, 500, 500, 900)
            self.draw_shapes(painter, 550, 750, 600, 900)
            self.draw_shapes(painter, 450, 750, 1100, 1500)
            self.draw_shapes(painter, 700, 1000, 1100, 1500)
            self.draw_shapes(painter, 900, 1000, 1100, 1500)
            self.draw_shapes(painter, 1000, 1200, 700, 1000)
            self.draw_shapes(painter, 1000, 1300, 550, 670)
            self.draw_shapes(painter, 1000, 1400, 50, 170)
            self.shapes_drawn = True
        else:
            for idx, shape in enumerate(polygons):
                if len(shape) > 2:
                    polygon_f = QPolygonF()# 用于绘制多边形
                    for point in shape:
                        polygon_f.append(point)
                    painter.setBrush(QBrush(QColor(245, 245, 220)))#米黄色
                    painter.setPen(QPen(QColor(0, 0, 0), 1))#外边框
                    painter.drawPolygon(polygon_f)
                    # 绘制顶点序号和标号点
                    """painter.setFont(QFont("Arial", 12, QFont.Bold))#字体
                    for i, pt in enumerate(shape):
                        painter.setPen(QPen(QColor(0, 128, 255), 2))# 设置画笔为蓝色，线宽2，用于绘制顶点圆圈
                        painter.setBrush(Qt.NoBrush)# 不填充
                        painter.drawEllipse(pt, 3,3)
                        painter.setPen(QPen(QColor(0, 0, 0), 1))#画编号的画笔
                        label = str(i + 1)
                        painter.drawText(int(pt.x()) + 5, int(pt.y()) - 5, label)"""

        # 画路径
        if self.start_set and self.end_set and self.path:
            painter.setPen(QPen(QColor(255, 0, 0), 2))#红色画笔
            for i in range(len(self.path) - 1):
                painter.drawLine(self.path[i], self.path[i + 1])
            # 绘制中间节点为绿色
            for i, pt in enumerate(self.path):
                if i == 0 or i == len(self.path) - 1:
                    continue
                painter.setBrush(QBrush(QColor(0, 200, 0)))#绿色
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(pt, 10, 10)
        if self.start_set:
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            painter.setPen(Qt.NoPen)#不绘制边框
            painter.drawEllipse(self.start, 12, 12)
        if self.end_set:
            painter.setBrush(QBrush(QColor(0, 0, 255)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.end, 12, 12)

    def draw_shapes(self, painter, min_x, max_x, min_y, max_y):
        global polygons, polygon, point_to_hull
        all_curves = []
        for j in range(4):
            control_points = []
            for i in range(4):
                control_points.append(QPointF(random.randint(min_x, max_x), random.randint(min_y, max_y)))
            smooth_curve = interpolate_bezier_closed(control_points, 2)
            unique_curve = []
            for point in smooth_curve:
                if point not in unique_curve:
                    unique_curve.append(point)
            all_curves.append(unique_curve)
        all_points = []
        for curve in all_curves:
            for point in curve:
                all_points.append(point)
        hull = convex_hull(all_points)
        hull_idx = len(polygons)
        for point in hull:
            polygon.append(point)
        polygons.append(hull)
        # 记录每个点属于哪个凸包
        for i, pt in enumerate(hull):
            idx = len(polygon) - len(hull) + i
            point_to_hull[idx] = hull_idx
        for curve in all_curves:
            if len(curve) > 2:
                polygon_f = QPolygonF()
                for point in curve:
                    polygon_f.append(point)
                painter.setBrush(QBrush(QColor(0, 255, 0)))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(polygon_f)
        if len(hull) > 1:
            polygon_f = QPolygonF()
            for point in hull:
                polygon_f.append(point)
            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            painter.drawPolygon(polygon_f)

    def is_point_in_any_polygon(self, point, min_distance=10):
        for poly in polygons:
            if self.point_in_polygon(point, poly):
                return True
        for poly in polygons:
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                distance = self.point_to_line_distance(point, p1, p2)
                if distance < min_distance:
                    return True
        return False

    def point_to_line_distance(self, point, line_start, line_end):
        line_vec_x = line_end.x() - line_start.x()
        line_vec_y = line_end.y() - line_start.y()
        point_vec_x = point.x() - line_start.x()
        point_vec_y = point.y() - line_start.y()
        line_len_squared = line_vec_x * line_vec_x + line_vec_y * line_vec_y
        if line_len_squared == 0:
            return math.hypot(point_vec_x, point_vec_y)
        t = max(0, min(1, (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_len_squared))
        proj_x = line_start.x() + t * line_vec_x
        proj_y = line_start.y() + t * line_vec_y
        dx = point.x() - proj_x
        dy = point.y() - proj_y
        return math.hypot(dx, dy)

    def mousePressEvent(self, event):
        if not self.drawing_area.geometry().contains(event.pos()):
            return
        if event.button() == Qt.LeftButton:
            click_position = event.pos()
            point_qf = QPointF(click_position)
            if self.is_point_in_any_polygon(point_qf):
                QMessageBox.warning(self, "无效选点", "无效选点，请重新选点")
                print("无效选点：点在障碍物内部")
                return
            self.points_history.append(click_position)
            if not self.start_set:
                self.start = QPointF(click_position)
                polygon.append(self.start)
                self.start_set = True
                print(f"起点设置在: ({self.start.x()}, {self.start.y()})")
                self.status_bar.showMessage("起点已设置，请点击设置终点")
            elif not self.end_set:
                self.end = QPointF(click_position)
                polygon.append(self.end)
                self.end_set = True
                print(f"终点设置在: ({self.end.x()}, {self.end.y()})")
                self.calculate_path()
            else:
                self.end = QPointF(click_position)
                if len(polygon) >= 2:
                    polygon[-1] = self.end
                else:
                    polygon.append(self.end)
                print(f"终点更新为: ({self.end.x()}, {self.end.y()})")
                self.calculate_path()
            self.update()

    def calculate_path(self):
        try:
            self.add_extra_points()
            start_idx = polygon.index(self.start)
            end_idx = polygon.index(self.end)
            se_indices = [start_idx, end_idx]
            distance_matrix = compute_distance_matrix(polygon, polygons, point_to_hull, se_indices)
            n = len(distance_matrix)
            next_matrix = [[-1 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if distance_matrix[i][j] != float('inf'):
                        next_matrix[i][j] = j
            update_distance_matrix(distance_matrix, next_matrix)
            self.path = construct_path(start_idx, end_idx, polygon, next_matrix)
            if self.path and len(self.path) > 1:
                path_length = self.calculate_path_length()
                self.status_bar.showMessage(f"已找到有效路径！路径长度: {path_length:.2f}")
                QMessageBox.information(self, "路径计算完成", f"已找到最短路径，最短路径是{path_length:.2f}")
            else:
                self.status_bar.showMessage("未找到有效路径，请尝试其他位置")
        except Exception as e:
            print("路径计算出现异常:", e)
            traceback.print_exc()
            self.status_bar.showMessage("路径计算异常，请检查控制台输出")

    def add_extra_points(self):
        for poly in polygons:
            for point in poly:
                if point not in polygon:
                    polygon.append(point)
        
    @staticmethod
    def point_in_polygon(point, polygon):
        n = len(polygon)
        inside = False
        if n < 3 or point is None:
            return False
        p1x, p1y = polygon[0].x(), polygon[0].y()
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x(), polygon[i % n].y()
            if point.y() > min(p1y, p2y):
                if point.y() <= max(p1y, p2y):
                    if point.x() <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y() - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x() <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def calculate_path_length(self):
        if not self.path or len(self.path) < 2:
            return 0
        length = 0
        for i in range(len(self.path) - 1):
            length += calculate_distance(self.path[i], self.path[i + 1])
        return length

    def reset(self):
        self.start = None
        self.end = None
        self.start_set = False
        self.end_set = False
        self.path = []
        self.points_history = []
        
        # 重置全局变量
        global polygon, point_to_hull, polygons
        polygon = []
        point_to_hull = {}
        
        # 重要：重新构建障碍物顶点与凸包的关联关系
        for hull_idx, hull in enumerate(polygons):
            for point in hull:
                if point not in polygon:
                    polygon.append(point)
                    # 记录每个点属于哪个凸包
                    point_to_hull[len(polygon) - 1] = hull_idx
        
        self.status_bar.showMessage("已重置，请点击设置起点")
        self.update()

    def undo(self):
        if not self.points_history:
            self.status_bar.showMessage("没有可撤销的操作")
            return
        self.points_history.pop()
        self.start = None
        self.end = None
        self.start_set = False
        self.end_set = False
        self.path = []
        
        # 重置全局变量
        global polygon, point_to_hull, polygons
        polygon = []
        point_to_hull = {}
        
        # 重要：重新构建障碍物顶点与凸包的关联关系
        for hull_idx, hull in enumerate(polygons):
            for point in hull:
                if point not in polygon:
                    polygon.append(point)
                    # 记录每个点属于哪个凸包
                    point_to_hull[len(polygon) - 1] = hull_idx
        
        # 重新添加起点和终点
        for i, point in enumerate(self.points_history):
            if i == 0:
                self.start = QPointF(point)
                polygon.append(self.start)
                self.start_set = True
                self.status_bar.showMessage("起点已设置，请点击设置终点")
            elif i == 1:
                self.end = QPointF(point)
                polygon.append(self.end)
                self.end_set = True
                self.calculate_path()
        
        if not self.points_history:
            self.status_bar.showMessage("请点击设置起点")
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PathPlanningApp()
    window.show()
    sys.exit(app.exec_())
