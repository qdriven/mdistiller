import os
import sys
import requests
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    """
    下载文件并显示进度条
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if response.status_code != 200:
        print(f"下载失败，状态码: {response.status_code}")
        return False
        
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    return True

def setup_teacher_checkpoints():
    """
    下载并解压教师模型检查点
    """
    # 创建下载目录
    download_dir = "download_ckpts"
    os.makedirs(download_dir, exist_ok=True)
    
    # 下载地址
    url = "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
    tar_file = os.path.join(download_dir, "cifar_teachers.tar")
    
    # 检查是否已经存在
    if os.path.exists(os.path.join(download_dir, "cifar_teachers")):
        print("检查点文件已存在，跳过下载")
        return True
    
    # 下载文件
    print("正在下载教师模型检查点...")
    if not download_file(url, tar_file):
        print("下载失败")
        return False
    
    # 解压文件
    print("正在解压文件...")
    try:
        with tarfile.open(tar_file) as tar:
            tar.extractall(download_dir)
        print("解压完成")
        
        # 删除tar文件
        os.remove(tar_file)
        print("清理完成")
        
        return True
    except Exception as e:
        print(f"解压失败: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_teacher_checkpoints():
        print("教师模型检查点设置成功！")
        print("检查点位置: ./download_ckpts/cifar_teachers/")
    else:
        print("设置失败")
        sys.exit(1)