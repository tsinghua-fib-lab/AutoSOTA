p = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'
c = open(p).read()
c = c.replace('w_max, w_min = 2.0, 1.0', 'w_max, w_min = 1.5, 1.0')
open(p, 'w').write(c)
# verify
for l in open(p):
    if 'w_max' in l and 'w_min' in l:
        print(l.strip())
