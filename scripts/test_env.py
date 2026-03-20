# test_env.py
try:
    from simple_knn._C import distCUDA2
    print('✓ simple-knn 模块正常')
    from xray_gaussian_rasterization_voxelization._C import rasterize_gaussians  
    print('✓ xray-gaussian 模块正常')
    print('环境就绪，可以开始训练！')
except Exception as e:
    print('✗ 模块导入失败:', e)