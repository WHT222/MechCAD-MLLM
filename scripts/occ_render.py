import os
import sys
import glob
import json
import h5py
import argparse

import numpy as np
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
import OCC.Core.V3d as V3d
import OCC.Core.Graphic3d as Graphic3d

def ensure_dir(path):
    """
    通过首次检查路径是否存在来创建路径
    :param path: 路径
    :return: 无
    """
    if not os.path.exists(path):
        os.makedirs(path)

display, start_display, add_menu, add_function_to_menu = init_display()
display.View.TriedronErase() #type: ignore

def generate_image_of_step_entity(
    step_file: str,
    save_dir: str,
):
    """
    为 STEP 实体生成图像。

    参数:
        step_file: STEP 文件的路径

    返回:
        None
    """
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(step_file)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    color = Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB) #type: ignore
    display.DisplayColoredShape(shape, color, update=True)#type: ignore
    # display.DisplayShape(shape, update=True)
    print(os.path.join(save_dir, step_file.split('/')[-1].split('.')[0]))

    proj = list(display.View.Proj())#type: ignore

    for i in range(8):
        count = i
        new_proj = proj.copy()
        if (i & 1) != 0:
            new_proj[0] *= -1
        if (i & 2) != 0:
            new_proj[1] *= -1
        if (i & 4) != 0:
            new_proj[2] *= -1
        display.View.SetProj(new_proj[0], new_proj[1], new_proj[2])#type: ignore
        display.View.Dump(os.path.join(save_dir, step_file.split('/')[-1].split('.')[0] + "_{:03d}.png".format(count)))#type: ignore

    display.EraseAll()#type: ignore
    display.View.Reset()#type: ignore


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help="源文件夹")
    parser.add_argument('--idx', type=int, default=0, help="从 idx 开始显示 n 个文件")
    parser.add_argument('--num', type=int, default=10, help="要显示的形状数量。-1 表示显示所有形状")
    parser.add_argument('--with_gt', action="store_true", help="同时显示真实值")
    parser.add_argument('--filter', action="store_true", help="使用 OpenCascade 分析器过滤无效模型")
    parser.add_argument('-o', '--outputs', type=str, default=None, help="保存文件夹")
    args = parser.parse_args()

    src_dir = args.src
    print(src_dir)

    out_paths = sorted(glob.glob(os.path.join(src_dir, "**", "*.step"), recursive=True))

    if args.num != -1:
        out_paths = out_paths[args.idx:args.idx + args.num]

    save_dir = args.src + "_img" if args.outputs is None else args.outputs
    ensure_dir(save_dir)

    for i in out_paths:
        generate_image_of_step_entity(i, save_dir)