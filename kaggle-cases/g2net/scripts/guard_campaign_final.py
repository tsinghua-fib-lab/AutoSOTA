"""Detached deadline guardian for the final campaign and its recorded children."""
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import run_campaign10h as e

ROOT=e.ROOT
DEADLINE=dt.datetime.fromisoformat('2026-10-14T10:00:00+08:00').timestamp()
STATE=ROOT/'reports/campaign_final_status.json'


def recorded_children():
    if not STATE.exists(): return []
    try: data=json.loads(STATE.read_text())
    except (OSError,ValueError): return []
    found=[]
    for entry in data.get('jobs',{}).values():
        pid=entry.get('pid')
        if not pid:continue
        try:
            actual=Path('/proc/%d/cmdline'%pid).read_bytes().split(b'\0')[:-1]
            expected=[str(arg).encode() for arg in entry['command']]
            if actual==expected and os.getpgid(pid)==pid:found.append(pid)
        except (OSError,KeyError):pass
    return found


def send(pid,sig):
    try:os.killpg(pid,sig)
    except ProcessLookupError:pass


def main():
    lock=(ROOT/'reports/campaign_final_guard.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if STATE.exists():raise FileExistsError('Refuse duplicate final campaign')
    with (ROOT/'logs/campaign_final_manager.log').open('w') as log:
        manager=subprocess.Popen([e.TPY,'scripts/run_campaign_final.py'],cwd=ROOT,env=e.ENV,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(guardian_pid=os.getpid(),manager_pid=manager.pid,deadline_china='2026-10-14T10:00:00+08:00',
              state='running',started_at=e.now())
    target=ROOT/'reports/campaign_final_guard.json';e.atomic_json(target,info)
    stopping=False
    def stop(*unused):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while manager.poll() is None and time.time()<DEADLINE-3 and not stopping:
        time.sleep(min(5,max(.1,DEADLINE-3-time.time())))
    children=recorded_children()
    if manager.poll() is None or children:
        for pid in children:send(pid,signal.SIGTERM)
        send(manager.pid,signal.SIGTERM)
        timeout=max(0,min(3,DEADLINE-time.time()))
        try:manager.wait(timeout=timeout)
        except subprocess.TimeoutExpired:send(manager.pid,signal.SIGKILL)
        for pid in children:
            # These groups were ownership-verified immediately above. Also
            # kill any remaining data-loader descendants in the same group.
            send(pid,signal.SIGKILL)
    code=manager.wait()
    info.update(state='deadline_stopped' if time.time()>=DEADLINE-3 else ('stopped' if stopping else 'manager_exited'),
                finished_at=e.now(),manager_exit_code=code)
    e.atomic_json(target,info)


if __name__=='__main__':main()
