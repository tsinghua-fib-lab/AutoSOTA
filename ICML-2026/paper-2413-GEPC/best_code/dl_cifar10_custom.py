import ssl, requests, os, shutil, tarfile
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter

class CustomSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # Also try with older protocol versions
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', CustomSSLAdapter())

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
print(f'Trying with custom SSL adapter: {url}')
try:
    resp = session.get(url, stream=True, timeout=120, verify=False)
    print(f'Status: {resp.status_code}, final URL: {resp.url[:100]}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')

# Also try with the proxy explicitly
session2 = requests.Session()
session2.proxies = {'http': 'http://172.17.0.1:17890', 'https': 'http://172.17.0.1:17890'}
session2.mount('https://', CustomSSLAdapter())
try:
    resp = session2.get(url, stream=True, timeout=120, verify=False)
    print(f'With explicit proxy: Status: {resp.status_code}')
except Exception as e:
    print(f'With explicit proxy: Error: {type(e).__name__}: {e}')
