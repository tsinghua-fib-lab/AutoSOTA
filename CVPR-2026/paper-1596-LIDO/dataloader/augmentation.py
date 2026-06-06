import numpy as np

SCALE_RATE = 0.05
JITTER_RATE = 0.3
DROP_RATE = 0.1

def random_scale(scan, rate=0.05):
    scale = np.random.uniform(1-rate, 1+rate)
    scan[:, (0, 1)] *= scale

    return scan

def global_rotation(scan):
    angle = np.deg2rad(np.random.random() * 360)
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, sin], [-sin, cos]])
    scan[:, :2] = np.dot(scan[:, :2], R)

    return scan

def random_jitter(scan, rate=0.3):
    jitter = np.clip(np.random.normal(0, rate, 2), rate, rate)
    scan[:, :2] += jitter

    return scan

def random_flip(scan):
    flip = np.random.choice(4, 1)
    if flip == 1:
        scan[:, 0] = - scan[:, 0]
    elif flip == 2:
        scan[:, 1] = - scan[:, 1]
    elif flip == 3:
        scan[:, :2] = - scan[:, :2]
    
    return scan

def random_drop(scan, label, rate=0.1):
    drop = int(len(scan) * rate)
    drop = np.random.randint(low=0, high=drop)
    to_drop = np.random.randint(low=0, high=len(scan)-1, size=drop)
    to_drop = np.unique(to_drop)
    scan = np.delete(scan, to_drop, axis=0)
    label = np.delete(scan, to_drop, axis=0)

    return scan, label

def augmentation(scan, label=None):
    scan = random_scale(scan, SCALE_RATE)
    scan = global_rotation(scan)
    #scan = random_jitter(scan, JITTER_RATE) # does not seem to be useful so far
    scan = random_flip(scan)
    #if np.random.random() < 0.1 and label is not None:
        #scan = random_drop(scan, label, DROP_RATE)

    return scan