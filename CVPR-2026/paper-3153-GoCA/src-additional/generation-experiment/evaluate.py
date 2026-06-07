from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

fid, _ = calculate_fid('outputs/fid/official-CFG', get_coco_fid_stats())
print(fid)