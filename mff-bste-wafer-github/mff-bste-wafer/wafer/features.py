import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .common import BRANCHES
from .data import validate_map


def ratio(a, b):
    return float(a / b) if b else 0.0


def concentration(values):
    s = np.sum(values)
    return float(np.sum((values / s) ** 2)) if s else 0.0


class Wafer:
    def __init__(self, a):
        self.a = a
        self.valid = a != 0
        self.defect = a == 2
        self.valid_count = int(self.valid.sum())
        self.defect_count = int(self.defect.sum())

    def spatial_grid(self):
        cy, cx = ndi.center_of_mass(self.valid)
        y, x = np.indices(self.a.shape)
        r = np.hypot(y - cy, x - cx)
        radius = max(float(r[self.valid].max()), 1)
        sector = np.floor((np.arctan2(y - cy, x - cx) % (2 * np.pi)) / (np.pi / 4)).astype(int) % 8
        return cy, cx, radius, r / radius, sector


def density(w, region):
    return ratio(np.count_nonzero(w.defect & region), np.count_nonzero(w.valid & region))


def spatial(w, settings):
    cy, cx, radius, r, sector = w.spatial_grid()
    dy, dx = np.where(w.defect)
    radii = r[w.defect]
    bin_id = np.minimum((r * 6).astype(int), 5)
    counts = np.bincount(bin_id[w.defect], minlength=6)
    probabilities = counts[counts > 0] / max(w.defect_count, 1)
    offset_y = (dy.mean() - cy) / radius if dy.size else 0.0
    offset_x = (dx.mean() - cx) / radius if dx.size else 0.0
    values = dict(defect_share=ratio(w.defect_count, w.valid_count), centroid_dy=offset_y, centroid_dx=offset_x,
                  centroid_distance=np.hypot(offset_x, offset_y), radial_mean=radii.mean() if radii.size else 0,
                  radial_std=radii.std() if radii.size else 0, radial_entropy=-np.sum(probabilities * np.log2(probabilities)))
    values.update({f"zone_{name}": density(w, np.minimum((r * 3).astype(int), 2) == i)
                   for i, name in enumerate(("center", "middle", "edge"))})
    values.update({f"ring_{i + 1}": density(w, bin_id == i) for i in range(6)})
    values.update({f"sector_{i + 1}": density(w, sector == i) for i in range(8)})
    values.update({f"quadrant_{i + 1}": density(w, sector // 2 == i) for i in range(4)})
    return values


def components(w):
    indexed, n = ndi.label(w.defect, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(indexed.ravel(), minlength=n + 1)[1:]
    return indexed, sizes


def geometry(w, settings):
    indexed, sizes = components(w)
    keys = ("components", "mean_area", "std_area", "largest_share", "small_share", "bbox_aspect", "bbox_fill",
            "centroid_row", "centroid_column", "axis_sin2", "axis_cos2", "eccentricity", "compactness", "perimeter", "spread")
    if not sizes.size:
        return dict.fromkeys(keys, 0.0)
    largest = indexed == (sizes.argmax() + 1)
    y, x = np.where(largest)
    major = minor = angle = 0.0
    if len(x) > 1:
        xy = np.column_stack((x, y)).astype(float)
        xy -= xy.mean(axis=0)
        axes, directions = np.linalg.eigh(xy.T @ xy / len(x))
        minor, major = axes
        if major - minor > 1e-12:
            angle = np.arctan2(directions[1, -1], directions[0, -1])
    padded = np.pad(largest, 1).astype(np.int8)
    perimeter = int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
    width, height = int(np.ptp(x) + 1), int(np.ptp(y) + 1)
    all_y, all_x = np.where(w.defect)
    values = (ratio(sizes.size, w.valid_count), ratio(sizes.mean(), w.valid_count), ratio(sizes.std(), w.valid_count),
              ratio(sizes.max(), w.defect_count), ratio(np.sum(sizes[sizes <= 2]), w.defect_count),
              ratio(width, width + height), ratio(sizes.max(), width * height),
              ratio(y.mean(), w.a.shape[0] - 1), ratio(x.mean(), w.a.shape[1] - 1),
              np.sin(2 * angle) if major - minor > 1e-12 else 0, np.cos(2 * angle) if major - minor > 1e-12 else 0,
              np.sqrt(max(0, 1 - minor / major)) if major else 0,
              ratio(4 * np.pi * sizes.max(), perimeter ** 2), ratio(perimeter, w.valid_count),
              ratio(np.hypot(all_y.std(), all_x.std()), np.hypot(*w.a.shape)))
    return dict(zip(keys, values))


@lru_cache(maxsize=16)
def ray_indices(grid, angles):
    y, x = np.indices((grid, grid), dtype=float) - (grid - 1) / 2
    half = int(np.ceil(grid / np.sqrt(2)))
    theta = np.linspace(0, np.pi, angles, endpoint=False)
    indices = [np.rint(x * np.cos(t) + y * np.sin(t)).astype(int).ravel() + half for t in theta]
    return indices, 2 * half + 1


def sinogram(w, settings):
    grid = settings["radon_grid"]
    # Scale the two axes by the same factor, then center on a square discrete grid.
    h, width = w.a.shape
    nh, nw = [min(grid, max(1, int(round(size * grid / max(h, width))))) for size in (h, width)]
    y = np.minimum(h - 1, ((np.arange(nh) + .5) * h / nh).astype(int))
    x = np.minimum(width - 1, ((np.arange(nw) + .5) * width / nw).astype(int))
    resized = w.a[np.ix_(y, x)]
    a = np.zeros((grid, grid), dtype=np.uint8)
    top, left = (grid - nh) // 2, (grid - nw) // 2
    a[top:top + nh, left:left + nw] = resized
    rays, length = ray_indices(grid, settings["angles"])
    valid = a.ravel() != 0
    defect = a.ravel() == 2
    return (np.array([np.bincount(r[defect], minlength=length) for r in rays]),
            np.array([np.bincount(r[valid], minlength=length) for r in rays]))


def radon(w, settings):
    defects, valid = sinogram(w, settings)
    energy = np.array([concentration(line) for line in defects])
    normalized = np.divide(defects, valid, where=valid != 0, out=np.zeros_like(defects, dtype=float))
    supported = normalized[valid > 0]
    dominant = int(energy.argmax())
    theta = dominant * np.pi / len(energy)
    peak = int(np.argmax(defects[dominant]))
    values = {f"energy_{i:02d}": e for i, e in enumerate(energy)}
    values.update(mean=supported.mean() if supported.size else 0, variance=supported.var() if supported.size else 0,
                  peak=supported.max(initial=0), peak_position=ratio(peak, defects.shape[1] - 1) if energy.max() else 0,
                  directionality=ratio(energy.max() - energy.min(), energy.max()), angle_concentration=concentration(energy),
                  dominant_sin2=np.sin(2 * theta) if energy.max() else 0, dominant_cos2=np.cos(2 * theta) if energy.max() else 0)
    return values


def texture(w, settings):
    kernel = np.ones((3, 3), dtype=np.int16)
    near = ndi.convolve(w.defect.astype(np.int16), kernel, mode="constant")
    support = ndi.convolve(w.valid.astype(np.int16), kernel, mode="constant")
    local = (near[w.valid] / support[w.valid])
    defect_neighbors = (near - w.defect)[w.defect]
    support_neighbors = (support - w.valid)[w.defect]
    values = dict(local_mean=local.mean(), local_std=local.std(), local_max=local.max(),
                  local_q90=np.quantile(local, .9), defect_neighbor_share=ratio(defect_neighbors.sum(), support_neighbors.sum()),
                  isolated_share=ratio(np.sum(defect_neighbors == 0), w.defect_count))
    pairs = (("horizontal", np.s_[:, :-1], np.s_[:, 1:]), ("vertical", np.s_[:-1, :], np.s_[1:, :]),
             ("diagonal_down", np.s_[:-1, :-1], np.s_[1:, 1:]), ("diagonal_up", np.s_[1:, :-1], np.s_[:-1, 1:]))
    for name, left, right in pairs:
        valid = w.valid[left] & w.valid[right]
        a, b = w.defect[left], w.defect[right]
        values[f"{name}_both"] = ratio(np.sum(a & b & valid), valid.sum())
        values[f"{name}_different"] = ratio(np.sum((a ^ b) & valid), valid.sum())
    return values


def projections(w):
    for axis in (1, 0):
        bad, support = w.defect.sum(axis=axis), w.valid.sum(axis=axis)
        densities = np.divide(bad, support, where=support > 0, out=np.zeros_like(bad, dtype=float))
        yield bad, support, densities


def projection(w, settings):
    values = {}
    for name, (bad, support, rates) in zip(("row", "column"), projections(w)):
        supported = rates[support > 0]
        peaks, _ = find_peaks(np.pad(rates, 1), prominence=.02)
        values.update({f"{name}_{key}": value for key, value in dict(max=supported.max(), mean=supported.mean(),
                      variance=supported.var(), count_max=ratio(bad.max(), w.defect_count), concentration=concentration(bad),
                      peak_share=ratio(len(peaks), len(supported)), zero_share=np.mean(supported == 0)).items()})
    values["max_difference"] = abs(values["row_max"] - values["column_max"])
    values["variance_difference"] = abs(values["row_variance"] - values["column_variance"])
    return values


EXTRACTORS = dict(zip(BRANCHES, (spatial, geometry, radon, texture, projection)))


def extract(a, settings, branches=BRANCHES, *, validated=False):
    start = time.perf_counter()
    w = Wafer(a if validated else validate_map(a))
    times = {"mask": time.perf_counter() - start}
    groups = OrderedDict()
    for branch in branches:
        start = time.perf_counter()
        groups[branch] = {f"{branch}.{key}": float(value) for key, value in EXTRACTORS[branch](w, settings).items()}
        times[branch] = time.perf_counter() - start
    return w, groups, times


def describe_feature(name):
    branch, key = name.split(".")
    if branch == "spatial":
        if key.startswith(("zone_", "ring_", "sector_", "quadrant_")):
            return "区域缺陷晶粒数 / 区域有效晶粒数；从有效晶粒质心分区", "[0, 1]", "比较缺陷集中在哪个半径或方向"
        if key.startswith("centroid_"):
            return "缺陷质心相对有效区质心的偏移或距离 / 最大有效半径", "[-1, 1]", "刻画缺陷整体偏心程度"
        if key == "radial_entropy":
            return "以 6 环缺陷计数的相对频率计算 Shannon 熵", "[0, log2(6)]", "刻画缺陷跨半径的分散程度"
        return {
            "defect_share": ("缺陷晶粒数 / 有效晶粒数", "[0, 1]", "全局缺陷面积占比"),
            "radial_mean": ("缺陷晶粒归一化半径的均值", "[0, 1]", "缺陷靠近中心或边缘的程度"),
            "radial_std": ("缺陷晶粒归一化半径的总体标准差", "[0, 0.5]", "缺陷沿半径的分散程度"),
        }[key]
    if branch == "geometry":
        return {
            "components": ("8 连通域数 / 有效晶粒数", "[0, 1]", "缺陷区域碎片化程度"),
            "mean_area": ("8 连通域面积均值 / 有效晶粒数", "[0, 1]", "典型团块规模"),
            "std_area": ("8 连通域面积总体标准差 / 有效晶粒数", "[0, 0.5]", "团块大小差异"),
            "largest_share": ("最大域晶粒数 / 缺陷晶粒数", "[0, 1]", "缺陷是否集中于单一团块"),
            "small_share": ("面积不超过 2 的域总晶粒数 / 缺陷晶粒数", "[0, 1]", "零散小缺陷比例"),
            "bbox_aspect": ("最大域包围盒宽 /（宽＋高）", "[0, 1]", "包围盒偏横或偏纵"),
            "bbox_fill": ("最大域晶粒数 / 包围盒面积", "[0, 1]", "团块对矩形包围盒的填充程度"),
            "centroid_row": ("最大域质心行坐标 /（图高−1）", "[0, 1]", "团块在图内的纵向位置"),
            "centroid_column": ("最大域质心列坐标 /（图宽−1）", "[0, 1]", "团块在图内的横向位置"),
            "axis_sin2": ("最大域坐标协方差主轴 θ 的 sin(2θ)", "[-1, 1]", "无首尾主轴方向的正弦分量"),
            "axis_cos2": ("最大域坐标协方差主轴 θ 的 cos(2θ)", "[-1, 1]", "无首尾主轴方向的余弦分量"),
            "eccentricity": ("sqrt(1−次特征值/主特征值)", "[0, 1]", "最大域的细长度"),
            "compactness": ("4π×最大域面积 / 四向网格边界长度²", "[0, 1]", "最大域的圆形紧致程度"),
            "perimeter": ("最大域四向网格边界长度 / 有效晶粒数", "[0, 4]", "团块边界相对长度"),
            "spread": ("全缺陷横纵坐标标准差的欧氏范数 / 图像对角长度", "[0, 0.5]", "全缺陷在图内的分散程度"),
        }[key]
    if branch == "radon":
        if key.startswith("energy_"):
            return "指定角度缺陷投影按总缺陷数归一化后求平方和", "[0, 1]", "该角度线积分的集中度"
        return {
            "mean": ("对有效投影格的缺陷数 / 有效晶粒数取均值", "[0, 1]", "不同投影格的平均缺陷量"),
            "variance": ("对有效投影格缺陷密度取总体方差", "[0, 0.25]", "各投影线的密度差异"),
            "peak": ("对所有角度的有效投影格取最大缺陷密度", "[0, 1]", "最强投影线的缺陷比例"),
            "peak_position": ("最大能量角度下的计数峰格 /（投影格数−1）", "[0, 1]", "主方向的峰值位置"),
            "directionality": ("（最大方向能量−最小方向能量）/ 最大方向能量", "[0, 1]", "不同角度的强度差异"),
            "angle_concentration": ("方向能量按总能量归一化后求平方和", "[0, 1]", "投影能量在角度间的集中度"),
            "dominant_sin2": ("最大能量角度 θ 的 sin(2θ)", "[-1, 1]", "主投影法线方向的正弦分量"),
            "dominant_cos2": ("最大能量角度 θ 的 cos(2θ)", "[-1, 1]", "主投影法线方向的余弦分量"),
        }[key]
    if branch == "texture":
        if key.endswith(("_both", "_different")):
            event = "均缺陷" if key.endswith("_both") else "一正常一缺陷"
            return f"指定方向{event}的晶粒对数 / 对应方向有效晶粒对数", "[0, 1]", "指定方向的相邻状态关系"
        return {
            "local_mean": ("有效中心的 3×3 邻域缺陷率均值", "[0, 1]", "典型局部缺陷密度"),
            "local_std": ("有效中心的 3×3 邻域缺陷率总体标准差", "[0, 0.5]", "局部密度波动"),
            "local_max": ("有效中心的 3×3 邻域缺陷率最大值", "[0, 1]", "局部缺陷热点强度"),
            "local_q90": ("有效中心的 3×3 邻域缺陷率 90% 分位数", "[0, 1]", "高密度局部区域的水平"),
            "defect_neighbor_share": ("缺陷中心的八邻域缺陷数之和 / 有效邻居数之和", "[0, 1]", "缺陷附近仍是缺陷的比例"),
            "isolated_share": ("没有缺陷邻居的缺陷晶粒数 / 缺陷晶粒数", "[0, 1]", "孤立缺陷比例"),
        }[key]
    if key.endswith("_difference"):
        measure = "最大密度" if key.startswith("max") else "密度方差"
        return f"行与列投影的{measure}之差的绝对值", "[0, 1]", "水平与垂直投影的差异"
    _, measure = key.split("_", 1)
    return {
        "max": ("对应轴各有效行/列缺陷密度的最大值", "[0, 1]", "最强条带的密度"),
        "mean": ("对应轴各有效行/列缺陷密度的均值", "[0, 1]", "该方向的平均缺陷密度"),
        "variance": ("对应轴各有效行/列缺陷密度的总体方差", "[0, 0.25]", "不同位置的密度差异"),
        "count_max": ("对应轴单行/列的最大缺陷晶粒数 / 总缺陷晶粒数", "[0, 1]", "最多缺陷是否挤在一条线上"),
        "concentration": ("对应轴各行/列的缺陷计数比例平方和", "[0, 1]", "投影缺陷计数集中度"),
        "peak_share": ("密度曲线上突显度至少 0.02 的峰数 / 有效行列数", "[0, 1]", "条带峰值在轴上的频率"),
        "zero_share": ("无缺陷的有效行/列数 / 有效行列数", "[0, 1]", "该方向上无缺陷区域的比例"),
    }[measure]
