import argparse
import os.path as osp
import pickle
import sys
from multiprocessing import Pool

import cv2
import lmdb


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util as util  # noqa: E402


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def create_dataset(name, img_folder, lmdb_save_path, H_dst, W_dst, C_dst):
    """Create lmdb for the dataset, each image with a fixed size
    key pattern: folder_frameid
    """
    # configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith(".lmdb"):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if osp.exists(lmdb_save_path):
        print("Folder [{:s}] already exists. Exit...".format(lmdb_save_path))
        sys.exit(1)

    # read all the image paths to a list
    print("Reading image path list ...")
    all_img_list = data_util._get_paths_from_images(img_folder)
    keys = []
    for img_path in all_img_list:
        split_rlt = img_path.split("/")
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split(".png")[0]
        keys.append(folder + "_" + img_name)

    if read_all_imgs:
        # read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print("Read images with multiprocessing, #thread: {} ...".format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            """get the image data and update pbar"""
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update("Reading {}".format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print("Finish reading {} images.\nWrite lmdb...".format(len(all_img_list)))

    # create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print("data size per image is: ", data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update("Write {}".format(key))
        key_byte = key.encode("ascii")
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)

        assert len(data.shape) > 2 or C_dst == 1, "different shape"

        if C_dst == 1:
            H, W = data.shape
            assert H == H_dst and W == W_dst, "different shape."
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, "different shape."
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print("Finish writing lmdb.")

    # create meta information
    meta_info = {}
    meta_info["name"] = name
    channel = C_dst
    meta_info["resolution"] = "{}_{}_{}".format(channel, H_dst, W_dst)
    meta_info["keys"] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, "meta_info.pkl"), "wb"))
    print("Finish creating lmdb meta info.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel extractor testing")

    parser.add_argument("--H", action="store", help="source image height", type=int, required=True)
    parser.add_argument("--W", action="store", help="source image height", type=int, required=True)
    parser.add_argument("--C", action="store", help="source image height", type=int, required=True)
    parser.add_argument("--img_folder", action="store", help="img folder", type=str, required=True)
    parser.add_argument("--save_path", action="store", help="save path", type=str, default=".")
    parser.add_argument("--name", action="store", help="dataset name", type=str, required=True)

    args = parser.parse_args()
    create_dataset(args.name, args.img_folder, args.save_path, args.H, args.W, args.C)
