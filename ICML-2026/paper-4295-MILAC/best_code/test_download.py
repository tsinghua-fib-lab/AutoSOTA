"""Quick test of Wikipedia pageview download."""
import time, numpy as np, requests, os
from datetime import datetime

HEADERS = {'User-Agent': 'MILCCI-Reproduction/1.0 (research@example.com)'}
START = '20201009'
END = '20201029'
T_DAYS = (datetime.strptime(END, '%Y%m%d') - datetime.strptime(START, '%Y%m%d')).days + 1

def fetch(project, access, agent, article):
    url = (f'https://wikimedia.org/api/rest_v1/metrics/pageviews/'
           f'per-article/{project}/{access}/{agent}/{article}/'
           f'daily/{START}/{END}')
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                items = r.json().get('items', [])
                views = np.zeros(T_DAYS)
                sd = datetime.strptime(START, '%Y%m%d')
                for item in items:
                    ts = item['timestamp'][:8]
                    d = datetime.strptime(ts, '%Y%m%d')
                    idx = (d - sd).days
                    if 0 <= idx < T_DAYS:
                        views[idx] = item['views']
                return views
            elif r.status_code == 404:
                return None
            else:
                time.sleep(1)
        except Exception as e:
            print(f'    exc: {e}')
            time.sleep(1)
    return None

pages = ['Machine_learning', 'Facebook', 'Classical_conditioning']
languages = {
    'en': 'en.wikipedia.org',
    'fr': 'fr.wikipedia.org',
    'zh': 'zh.wikipedia.org',
}

for page in pages:
    print(f'--- {page} ---')
    for lc, proj in languages.items():
        for agent in ['user', 'spider']:
            plats = ['desktop', 'mobile-web', 'mobile-app'] if agent == 'user' else ['desktop', 'mobile-web']
            for plat in plats:
                v = fetch(proj, plat, agent, page)
                total = int(v.sum()) if v is not None else -1
                nonzero = int(np.count_nonzero(v)) if v is not None else -1
                print(f'  {lc}/{agent}/{plat}: total={total}, nonzero_days={nonzero}')
                time.sleep(0.15)
print('Done!')
