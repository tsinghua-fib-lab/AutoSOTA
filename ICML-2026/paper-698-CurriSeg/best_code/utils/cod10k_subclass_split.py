import os
import shutil


def split_GT():
    src_root = ''
    dst_root = ''

    for img_name in os.listdir(src_root):
        sub_class = img_name.split('-')[3]

        src_img_path = os.path.join(src_root, img_name)
        dst_img_path = os.path.join(dst_root, 'COD10K-'+sub_class, "GT", img_name)
        os.makedirs(os.path.join(dst_root, 'COD10K-'+sub_class, "GT"), exist_ok=True)

        shutil.copyfile(src_img_path, dst_img_path)


def split_pred():
    src_root = ''
    dst_root = ''

    for img_name in os.listdir(src_root):
        sub_class = img_name.split('-')[3]

        src_img_path = os.path.join(src_root, img_name)
        dst_img_path = os.path.join(dst_root, 'COD10K-' + sub_class, img_name)
        os.makedirs(os.path.join(dst_root, 'COD10K-' + sub_class), exist_ok=True)

        shutil.copyfile(src_img_path, dst_img_path)


split_pred()