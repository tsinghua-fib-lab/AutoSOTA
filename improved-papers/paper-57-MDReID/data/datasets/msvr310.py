# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""


import os
import os.path as osp
from .bases import BaseImageDataset


class MSVR310(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'MSVR310'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(MSVR310, self).__init__()
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query3')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        # pdb.set_trace()
        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        # pdb.set_trace()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for vid in os.listdir(dir_path):
            vid_path = osp.join(dir_path, vid)
            r_data = os.listdir(osp.join(vid_path, 'vis'))
            for img in r_data:
                r_img_path = osp.join(vid_path, 'vis', img)
                n_img_path = osp.join(vid_path, 'ni', img)
                t_img_path = osp.join(vid_path, 'th', img)
                vid = int(img[0:4])
                camid = int(img[11])
                sceneid = int(img[6:9])  # scene id
                assert 0 <= camid <= 7
                if relabel:
                    vid = vid2label[vid]
                dataset.append(((r_img_path, n_img_path, t_img_path), vid, camid, sceneid))
        return dataset

    # def _process_dir2(self, dir_path, relabel=False):
    #     img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    #     pattern = re.compile(r'([-\d]+)_c([-\d]+)')

    #     pid_container = set()
    #     for img_path in img_paths:
    #         pid, _ = map(int, pattern.search(img_path).groups())
    #         if pid == -1: continue  # junk images are just ignored
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}

    #     dataset = []
    #     for img_path in img_paths:
    #         pid, camid = map(int, pattern.search(img_path).groups())
    #         #pdb.set_trace()
    #         #if pid == -1: continue  # junk images are just ignored
    #         assert 1 <= pid <= 600  # pid == 0 means background
    #         assert 1 <= camid <= 8
    #         camid -= 1  # index starts from 0
    #         if relabel: pid = pid2label[pid]
    #         dataset.append((img_path, pid, camid))
    #     return dataset
