import torch
from scene.cameras import Camera

def rotation_matrix_to_quaternion_torch(R):
    assert R.shape == (3, 3)

    # 计算四元数的各个分量
    q_w = torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q_x = (R[2, 1] - R[1, 2]) / (4 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4 * q_w)

    return torch.tensor([q_w, q_x, q_y, q_z], device=R.device)

def quaternion_to_rotation_matrix_torch(q):
    # 四元数分量
    q_w, q_x, q_y, q_z = q

    # 计算旋转矩阵
    R = torch.tensor([
        [1 - 2 * q_y ** 2 - 2 * q_z ** 2, 2 * q_x * q_y - 2 * q_z * q_w, 2 * q_x * q_z + 2 * q_y * q_w],
        [2 * q_x * q_y + 2 * q_z * q_w, 1 - 2 * q_x ** 2 - 2 * q_z ** 2, 2 * q_y * q_z - 2 * q_x * q_w],
        [2 * q_x * q_z - 2 * q_y * q_w, 2 * q_y * q_z + 2 * q_x * q_w, 1 - 2 * q_x ** 2 - 2 * q_y ** 2]
    ], device=q.device)

    return R

def extend_cameras(cameras, num=6):
    cameras_extend = []
    for i in range(len(cameras) - 1):
        camera0 = cameras[i]
        camera1 = cameras[i+1]
        # cameras_extend.append(camera0)

        for j in range(1, num):
            # 插值平移矩阵
            T = (camera1.T - camera0.T) * j / num + camera0.T
            
            # 插值旋转矩阵
            R0 = camera0.R
            R1 = camera1.R
            q0 = rotation_matrix_to_quaternion_torch(R0)
            q1 = rotation_matrix_to_quaternion_torch(R1)
            q = (q1 - q0) * j / num + q0
            R = quaternion_to_rotation_matrix_torch(q)

            # 创建新插值相机
            cam = Camera(
                colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(),
                FoVx=camera0.FoVx, FoVy=camera0.FoVy,
                image=camera0.original_image, gt_alpha_mask=camera0.gt_alpha_mask,
                image_name=camera0.image_name, uid=camera0.uid,
                data_device=camera0.data_device,
                trans=(camera0.trans), scale=camera0.scale,
                HWK=camera0.HWK, gt_refl_mask=camera0.refl_mask
            )
            cameras_extend.append(cam)

    cameras_extend.append(cameras[-1])
    return cameras_extend
