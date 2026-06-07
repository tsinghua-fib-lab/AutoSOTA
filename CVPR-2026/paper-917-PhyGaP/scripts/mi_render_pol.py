import sys
sys.path.append("/root/autodl-tmp/mitsuba3/build/python")
import mitsuba as mi
mi.set_variant('cuda_ad_spectral_polarized')
import matplotlib.pyplot as plt
import numpy as np
from mitsuba import ScalarTransform4f as T
import cv2
import os # 【新增代码】 导入os模块来创建目录
def color_and_norm_dop_aolp(aolp, dop,channel = 1):
    import matplotlib.cm as cm
    vmin, vmax = 0, 180
    # aolp, dop are H, W, C; average over channel dimension
    if aolp.ndim == 3:
        aolp = aolp[...,channel]#np.mean(aolp, axis=-1)
    if dop.ndim == 3:
        dop = dop[...,channel]# np.mean(dop, axis=-1)
    aolp_norm = np.clip((aolp - vmin) / (vmax - vmin), 0, 1)
    dop_norm = np.clip(dop, 0, 1)
    cmap_aolp = cm.get_cmap('twilight')
    cmap_dop = cm.get_cmap('viridis')
    aolp_colored = cmap_aolp(aolp_norm)[..., :3]
    dop_colored = cmap_dop(dop_norm)[..., :3]
    return aolp_colored, dop_colored
def calc_aolp_dop(stokes,mask=None):
    if mask is not None:
        mask = (mask[...,:] > 1e-5)
        stokes_masked = np.stack([stokes[...,0]*mask,stokes[...,1]*mask,stokes[...,2]*mask])
    cues = cues_from_stokes(stokes)
    aolp = cues['aolp']
    dop = cues['dop']
    return aolp, dop


def cues_from_stokes(stokes):
    sqrt, atan2 = np.sqrt, np.arctan2
    dop = sqrt((stokes[...,1:]**2).sum(-1))/stokes[...,0] # sqrt(S1^2+S2^2+S3^2)/S0 S3=0
    dop[stokes[...,0]<1e-6] = 0.
    aolp = 0.5*atan2(stokes[...,2],stokes[...,1])
    aolp = (aolp%np.pi)/np.pi*180  #转化为弧度
    s0 = stokes[...,0]
    # aolp = (aolp[...,0] * s0[...,0] + aolp[...,1] * s0[...,1] + aolp[...,2] * s0[...,2]) / s0.sum(axis=-1)
    # dop = (dop[...,0] * s0[...,0] + dop[...,1] * s0[...,1] + dop[...,2] * s0[...,2]) / s0.sum(axis=-1)
    return {'dop':dop,
            'aolp':aolp,
            's0':s0}

def get_intrinsics(fov_deg, width, height):
    """
    从FOV(垂直视角)和图像尺寸计算相机内参矩阵 K。
    
    Args:
        fov_deg (float): 相机的垂直视场角 (degrees).
        width (int): 图像宽度 (pixels).
        height (int): 图像高度 (pixels).

    Returns:
        np.ndarray: 3x3 的内参矩阵 K.
    """
    # 将FOV从角度转换为弧度
    fov_rad = np.deg2rad(fov_deg)
    
    # 根据FOV和高度计算焦距 (fy)
    # tan(fov/2) = (height/2) / fy
    fy = (height / 2.0) / np.tan(fov_rad / 2.0)
    fx = fy # 通常假设像素是正方形的，所以 fx = fy
    
    # 主点 (principal point) 通常在图像中心
    cx = width / 2.0
    cy = height / 2.0
    
    # 构建内参矩阵 K
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K

def get_camera_extrinsics(origin, target, up):
    """
    计算相机的外参矩阵 (cam_to_world)。
    这是一个标准的 look-at 实现。
    
    Args:
        origin (np.ndarray): 相机在世界坐标系中的位置.
        target (np.ndarray): 相机观察的目标点.
        up (np.ndarray): 定义相机“上”方向的向量.

    Returns:
        np.ndarray: 4x4 的相机到世界坐标系的变换矩阵.
    """
    # 1. 计算Z轴 (前向)
    if up[2] == 1:
        print("Warning: Using up vector [0, 0, 1], adjusting origin and target coordinates.")
        origin = np.array([origin[0], -origin[2], origin[1]])
        target = np.array([target[0], -target[2], target[1]])
        up = np.array([up[0], -up[2], up[1]])
    forward = np.array(target) - np.array(origin)
    forward = forward / np.linalg.norm(forward)

    # 2. 计算X轴 (右向)
    right = np.cross(forward, np.array(up))
    if np.linalg.norm(right) < 1e-8: # 防止 forward 和 up 共线
        # 如果共线，选择一个备用 up 向量
        if np.abs(forward[1]) > 0.99:
            up_alt = [1, 0, 0]
        else:
            up_alt = [0, 1, 0]
        right = np.cross(forward, up_alt)

    right = right / np.linalg.norm(right)

    # 3. 计算Y轴 (真正的上方向)
    true_up = np.cross(right, forward)

    # 4. 构建 4x4 变换矩阵 (OpenGL / Mitsuba 约定)
    cam_to_world = np.identity(4)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = true_up
    cam_to_world[:3, 2] = -forward
    cam_to_world[:3, 3] = origin
    gl_to_cv_transform = np.array([
    [1, 0,  0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0,  0, 1]
    ])
    # # import pdb;pdb.set_trace()
    cam_to_world =  cam_to_world @ gl_to_cv_transform
    # 返回世界到相机的变换矩阵（w2c），即c2w的逆
    cam_to_world_inv = np.linalg.inv(cam_to_world)
    return cam_to_world_inv

def load_sensor(r, phi, theta, target, film_width, film_height, fov):
    """
    创建 Mitsuba 传感器并返回其外参矩阵。
    """
    # 1. 计算相机在世界坐标系中的位置
    # Mitsuba 的变换链是从右向左应用的
    up = [0., 1. , 0.]
    origin_transform = T().rotate(up, phi).rotate([0, 0, 1], theta)
    origin = origin_transform @ mi.ScalarPoint3f([r, 0, 0])
    # import pdb;pdb.set_trace()
    # 2. 创建 Mitsuba 传感器
    mitsuba_sensor = mi.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': T().look_at(
            origin=origin,
            target=target,
            up=up
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 6000
        },
        'film': {
            'type': 'hdrfilm',
            'width': film_width,
            'height': film_height,
            'rfilter': {'type': 'tent'},
            'pixel_format': 'rgb',
        },
    })

    # 3. 使用我们的函数计算并返回外参矩阵，以确保与保存的格式一致
    extrinsic_matrix = get_camera_extrinsics(
        origin=np.array(origin),
        target=np.array(target),
        up=np.array(up)
    )
    
    return mitsuba_sensor, extrinsic_matrix

def get_sensors_circle(number_of_sensors, radius, theta, target, film_width, film_height, fov):
    """
    在一个圆形轨迹上生成一系列传感器。
    """
    sensors = []
    extrinsics = []
    for i in range(number_of_sensors):
        phi = i * (360.0 / number_of_sensors)
        sensor, extrinsic = load_sensor(radius, phi, theta, target, film_width, film_height, fov)
        sensors.append(sensor)
        print(i,extrinsic)
        extrinsics.append(extrinsic)
    return sensors, extrinsics


scene = mi.load_file('scenes/ball/scene.xml')

# 相机和输出设置
n_sensors = 36
radius = 10
theta = -50
target_point = [0,0,0]
film_width = 612
film_height = 512
fov_degrees = 28.8415
save_root = "ball" # 保存参数的目录

# 2. 生成所有传感器及其外参
sensors, opengl_extrinsics_list = get_sensors_circle(
    n_sensors, radius, theta, target_point, film_width, film_height, fov_degrees
)
# import pdb;pdb.set_trace()
# 3. 计算内参
# 对于这个设置，所有相机的内参都是相同的
intrinsic_matrix = get_intrinsics(fov_degrees, film_width, film_height)

# 4. 【关键步骤】将OpenGL/Mitsuba坐标系转换为COLMAP/OpenCV坐标系
# 这个变换矩阵将Y和Z轴反向，相当于绕X轴旋转180度。
# OpenGL (Y up, -Z forward) -> OpenCV (Y down, +Z forward)

colmap_extrinsics_list = opengl_extrinsics_list
print(colmap_extrinsics_list)
# import pdb;pdb.set_trace()
# 5. 准备保存数据
# 将转换后的外参矩阵列表堆叠成一个 NumPy 数组
extrinsic_matrices_stack = np.stack(colmap_extrinsics_list, axis=0)

# 确保保存目录存在
os.makedirs(save_root, exist_ok=True)
save_path = os.path.join(save_root, 'camera_params.npz')

# 6. 使用 np.savez 保存所有参数到一个文件中
np.savez(
    save_path,
    intrinsics=intrinsic_matrix,
    extrinsics=extrinsic_matrices_stack,
    image_width=film_width,
    image_height=film_height,
    fov=fov_degrees
)

# 7. 打印确认信息
print(f"成功保存 {n_sensors} 个相机的参数到: {save_path}")
print("这些参数现在与 COLMAP / OpenCV 的坐标系兼容。")
print("-" * 30)
print(f"文件内容:")
print(f"  - 'intrinsics' (内参 K), shape: {intrinsic_matrix.shape}")
print(f"  - 'extrinsics' (外参 [R|t]), shape: {extrinsic_matrices_stack.shape}")
print(f"  - 'image_width', 'image_height', 'fov'")
print("-" * 30)

images = [mi.render(scene, spp=1024, sensor=sensor) for sensor in sensors]

for idx, image in enumerate(images):
    bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())
    # 确保目录存在
    # import pdb; pdb.set_trace()
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(f"./{save_root}/viz_srgb/", exist_ok=True)
    os.makedirs(f"./{save_root}/viz_linear/", exist_ok=True)
    os.makedirs(f"./{save_root}/viz_S1/", exist_ok=True)
    os.makedirs(f"./{save_root}/viz_S2/", exist_ok=True)
    os.makedirs(f"./{save_root}/images/", exist_ok=True)
    # 将channels['s0']从sRGB空间转换到线性空间
    s0_srgb = np.array(channels['S0'])
    # sRGB转线性空间
    s0_linear = np.where(s0_srgb <= 0.04045, 
                        s0_srgb / 12.92, 
                        ((s0_srgb + 0.055) / 1.055) ** 2.4)

    for s_name in ['S1', 'S2']:
        channel_data = np.array(channels[s_name])
        # 分离RGB三个通道并分别保存
        # for c_idx, c_name in enumerate(['R', 'G', 'B']):
        #     channel_c = channel_data[..., c_idx]
        #     channel_c_linear = np.where(channel_c <= 0.04045, 
        #                         channel_c / 12.92, 
        #                         ((channel_c + 0.055) / 1.055) ** 2.4)
        #     mi.util.write_bitmap(f"./{save_root}/viz_{s_name}/{idx:03d}_{s_name}_{c_name}_bin.png", channel_c_linear)
        channel_linear = np.where(channel_data <= 0.04045,
                                channel_data / 12.92,
                                ((channel_data + 0.055) / 1.055) ** 2.4)
        
        mask_nonzero = channel_linear > 0
        channel_linear_bin = np.zeros_like(channel_linear)
        if np.any(mask_nonzero):
            min_val = channel_linear[mask_nonzero].min()
            max_val = channel_linear[mask_nonzero].max()
            channel_linear_bin[mask_nonzero] = (channel_linear[mask_nonzero] - min_val) / (max_val - min_val + 1e-8)
        mi.util.write_bitmap(f"./{save_root}/viz_{s_name}/{idx:03d}_{s_name}_bin.png", channel_linear_bin)
    # 将s0,s1,s2 stack后输出calc_aolp_dop，得到aolp,dop后color_and_norm_dop_aolp, 然后visualize normalize后的 aolp和dop
    s0 = np.array(channels['S0'])
    s1 = np.array(channels['S1'])
    s2 = np.array(channels['S2'])
    stokes = np.stack([s0, s1, s2], axis=-1)
    aolp, dop = calc_aolp_dop(stokes)
    aolp_colored_R, dop_colored_R = color_and_norm_dop_aolp(aolp, dop, channel = 0)
    aolp_colored_G, dop_colored_G = color_and_norm_dop_aolp(aolp, dop, channel = 1)
    aolp_colored_B, dop_colored_B = color_and_norm_dop_aolp(aolp, dop, channel = 2)
    # 将三个通道的
    aolp_img_R = (aolp_colored_R * 255).astype(np.uint8)
    aolp_img_G = (aolp_colored_G * 255).astype(np.uint8)
    aolp_img_B = (aolp_colored_B * 255).astype(np.uint8)
    dop_img_R = (dop_colored_R * 255).astype(np.uint8)
    dop_img_G = (dop_colored_G * 255).astype(np.uint8)
    dop_img_B = (dop_colored_B * 255).astype(np.uint8)
    # 保存三个通道的aolp和dop
    os.makedirs(f"./{save_root}/viz_aolp/", exist_ok=True)
    os.makedirs(f"./{save_root}/viz_dop/", exist_ok=True)
    mi.util.write_bitmap(f"./{save_root}/viz_aolp/{idx:03d}_aolp_R.png", aolp_img_R)
    mi.util.write_bitmap(f"./{save_root}/viz_aolp/{idx:03d}_aolp_G.png", aolp_img_G)
    mi.util.write_bitmap(f"./{save_root}/viz_aolp/{idx:03d}_aolp_B.png", aolp_img_B)
    mi.util.write_bitmap(f"./{save_root}/viz_dop/{idx:03d}_dop_R.png", dop_img_R)
    mi.util.write_bitmap(f"./{save_root}/viz_dop/{idx:03d}_dop_G.png", dop_img_G)
    mi.util.write_bitmap(f"./{save_root}/viz_dop/{idx:03d}_dop_B.png", dop_img_B)
    # # 保存 aolp_colored 和 dop_colored 为 PNG，归一化到 [0,255] 并转换为 uint8
    # aolp_img = (aolp_colored * 255).astype(np.uint8)
    # dop_img = (dop_colored * 255).astype(np.uint8)
    # # import pdb;pdb.set_trace()
    # os.makedirs(f"./{save_root}/viz_aolp/", exist_ok=True)
    # os.makedirs(f"./{save_root}/viz_dop/", exist_ok=True)
    # plt.imsave(f"./{save_root}/viz_aolp/{idx:03d}_aolp.png", aolp_img)
    # plt.imsave(f"./{save_root}/viz_dop/{idx:03d}_dop.png", dop_img)
    
    
    mi.util.write_bitmap(f"./{save_root}/viz_linear/{idx:03d}.png", s0_linear)
    mi.util.write_bitmap(f"./{save_root}/viz_srgb/{idx:03d}.png", s0_srgb)
    bitmap.write(f'./{save_root}/images/{idx:03d}.exr')


# import pdb;pdb.set_trace()