import os
import subprocess
import sys

def test_single_dataset():
    """先测试一个数据集确保环境正常"""
    
    # 测试一个简单的数据集
    test_datasets = [
        {
            "name": "测试数据集",
            "path": "data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone",
            "output": "output/test/chest"
        }
    ]
    
    common_args = [
        "--iterations", "1000",  # 先用较少的迭代测试
        "--batch_size", "1",
        "--rasterize_mode", "xray", 
        "--voxel_size", "0.01"
    ]
    
    for dataset in test_datasets:
        if not os.path.exists(dataset["path"]):
            print(f"数据集不存在: {dataset['path']}")
            continue
            
        print(f"\n开始测试: {dataset['name']}")
        print(f"数据路径: {dataset['path']}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(dataset["output"]), exist_ok=True)
        
        cmd = [
            "python", "train.py",
            "-s", dataset["path"],
            "-m", dataset["output"]
        ] + common_args
        
        try:
            print("执行命令:", " ".join(cmd))
            result = subprocess.run(cmd, check=True, text=True)
            print(f"✓ 测试成功: {dataset['name']}")
            break  # 如果成功就继续完整训练
        except subprocess.CalledProcessError as e:
            print(f"✗ 测试失败: {dataset['name']}")
            print(f"错误信息: {e}")
            if "DLL" in str(e) or "import" in str(e):
                print("\n需要先解决编译问题！")
                sys.exit(1)
        except Exception as e:
            print(f"✗ 异常: {str(e)}")
            sys.exit(1)

def main():
    print("开始批量训练...")
    
    # 先测试一个数据集
    test_single_dataset()
    
    # 如果测试成功，运行完整批量训练
    print("\n测试成功！开始完整批量训练...")
    
    # 这里可以调用您原来的批量训练函数
    # batch_train_full()

if __name__ == "__main__":
    main()