# 通用工具函数，后续拓展放入此处
import os

def create_output_dir(path="output"):
    if not os.path.exists(path):
        os.mkdir(path)
