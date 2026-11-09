"""Extract official competition archive, and inventory its inputs."""
from pathlib import Path
import json
import zipfile
import time

root = Path(__file__).resolve().parents[1]
src = root/'data/raw/google-smartphone-decimeter-challenge.zip'
dst = root/'data/competition'
start = time.time()
with zipfile.ZipFile(src) as z:
    infos = z.infolist()
    for info in infos:
        target = (dst/info.filename).resolve()
        if not target.is_relative_to(dst.resolve()):
            raise ValueError('Archive path outside destination')
        if not target.is_file() or target.stat().st_size != info.file_size:
            z.extract(info, dst)
    manifest = [{'name':i.filename,'bytes':i.file_size,'crc':i.CRC} for i in infos]
(root/'research/data_inventory.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'archive_files':len(infos),'uncompressed_bytes':sum(i.file_size for i in infos),'seconds':time.time()-start}), flush=True)
