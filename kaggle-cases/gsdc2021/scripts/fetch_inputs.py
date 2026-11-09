"""Download official competition data and the public smartphone auxiliary release."""
from pathlib import Path
import hashlib,shutil,subprocess,sys,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    cli=Path(sys.executable).parent/'kaggle'
    command=str(cli) if cli.exists() else shutil.which('kaggle')
    if not command:raise SystemExit('Install the dependency environment before downloading.')
    competition=raw/'google-smartphone-decimeter-challenge.zip'
    if not competition.exists():subprocess.run([command,'competitions','download','-c','google-smartphone-decimeter-challenge','-p',str(raw)],check=True)
    h=hashlib.sha256()
    with competition.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    if h.hexdigest()!='8edf2724305b36f986f29ef7f3cec99a854471fe0324675076f82b3bb0db1657':raise SystemExit('Downloaded official archive differs from the historical input; inspect before proceeding.')
    upstream=ROOT/'upstream';upstream.mkdir(exist_ok=True);archive=upstream/'demo5_b34e_phone.zip'
    if not archive.exists():
        with urllib.request.urlopen('https://github.com/rtklibexplorer/RTKLIB/releases/download/b34e_smartphone/demo5_b34e_phone.zip') as r,archive.open('wb') as out:shutil.copyfileobj(r,out)
    destination=upstream/'demo5_b34e_phone';destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target=(destination/info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive member')
            if not info.is_dir():z.extract(info,destination)
    print('External inputs downloaded. Next: python prepare_best.py --from-raw')

if __name__=='__main__':main()
