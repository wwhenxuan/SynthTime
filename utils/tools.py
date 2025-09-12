# -*- coding: utf-8 -*-
"""
Created on 2025/03/07 11:12:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import os


def makedir(directory: str, folder_name: str) -> None:
    """在指定目录下创建文件夹的函数"""
    # 构建完整的路径
    new_folder_path = os.path.join(directory, folder_name)

    # 判断目录是否存在，不存在则创建
    try:
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
    except OSError as e:
        print(f"Something went wrong when creating the directory: {e}")
