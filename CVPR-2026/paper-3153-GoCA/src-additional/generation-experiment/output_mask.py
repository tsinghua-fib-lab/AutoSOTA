import os
import numpy as np
from PIL import Image

palette = [
	[0.4420,  0.5100 , 0.4234],
	[0.8562,  0.9537 , 0.3188],
	[0.2405,  0.4699 , 0.9918],
	[0.8434,  0.9329  ,0.7544],
	[0.3748,  0.7917 , 0.3256],
	[0.0190,  0.4943 , 0.3782],
	[0.7461 , 0.0137 , 0.5684],
	[0.1644,  0.2402 , 0.7324],
	[0.0200 , 0.4379 , 0.4100],
	[0.5853 , 0.8880 , 0.6137],
	[0.7991 , 0.9132 , 0.9720],
	[0.6816 , 0.6237  ,0.8562],
	[0.9981 , 0.4692 , 0.3849],
	[0.5351 , 0.8242 , 0.2731],
	[0.1747 , 0.3626 , 0.8345],
	[0.5323 , 0.6668 , 0.4922],
	[0.2122 , 0.3483 , 0.4707],
	[0.6844,  0.1238 , 0.1452],
	[0.3882 , 0.4664 , 0.1003],
	[0.2296,  0.0401 , 0.3030],
	[0.5751 , 0.5467 , 0.9835],
	[0.1308 , 0.9628,  0.0777],
	[0.2849  ,0.1846 , 0.2625],
	[0.9764 , 0.9420 , 0.6628],
	[0.3893 , 0.4456 , 0.6433],
	[0.8705 , 0.3957 , 0.0963],
	[0.6117 , 0.9702 , 0.0247],
	[0.3668 , 0.6694 , 0.3117],
	[0.6451 , 0.7302,  0.9542],
	[0.6171 , 0.1097,  0.9053],
	[0.3377 , 0.4950,  0.7284],
	[0.1655,  0.9254,  0.6557],
	[0.9450  ,0.6721,  0.6162]
]
palette = (np.array(palette)*255).astype(np.uint8)

counter = 0
save_path = 'outputs/mask'
def output_mask(posi_mask, fore_mask):
	global counter
	img = palette[posi_mask]
	img = Image.fromarray(img)
	os.makedirs(save_path, exist_ok=True)
	img.save(os.path.join(save_path, f'posi_mask_{counter}.png'))

	img = palette[fore_mask]
	img = Image.fromarray(img)
	os.makedirs(save_path, exist_ok=True)
	img.save(os.path.join(save_path, f'fore_mask_{counter}.png'))

	palette_img = palette[:posi_mask.shape[0]+1]
	palette_img = palette_img.reshape((1,)+palette_img.shape)
	palette_img = Image.fromarray(palette_img).resize((512, 512), Image.NEAREST)
	palette_img.save(os.path.join(save_path, 'palette.png'))

	counter += 1
