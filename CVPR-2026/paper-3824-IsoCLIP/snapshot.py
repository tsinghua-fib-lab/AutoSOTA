"""Git snapshot before iteration."""
import subprocess, os
GIT_BIN = '/repo/git_bin'
GIT_ENV = {**os.environ, 'LD_LIBRARY_PATH': '/repo:' + os.environ.get('LD_LIBRARY_PATH', '')}

def git(*args):
    r = subprocess.run([GIT_BIN] + list(args), cwd='/repo', env=GIT_ENV, capture_output=True, text=True, timeout=30)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

git('config', '--global', '--add', 'safe.directory', '/repo')
git('config', 'user.name', 'optimizer')
git('config', 'user.email', 'opt@local')
git('add', '-A')
stdout, stderr, rc = git('commit', '-q', '-m', 'pre-iter-1: Soft Sigmoid Spectral Thresholding', '--allow-empty')
stdout, _, _ = git('rev-parse', 'HEAD')
git('tag', '-f', '_pre_iter')

with open('/repo/snapshot_log.txt', 'w') as f:
    f.write(f"PRE_COMMIT={stdout}\n")
    f.write(f"commit rc={rc}\n")
    f.write("DONE\n")
