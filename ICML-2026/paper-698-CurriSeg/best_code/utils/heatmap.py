import numpy as np
import cv2
import torch


def heatmap(x_show, img, name=None):
    x_show = torch.mean(x_show, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)

    img = img.data.cpu().numpy().squeeze()
    img = img.transpose((1, 2, 0))
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img[:, :, ::-1]
    img = np.uint8(255 * img)
    x_show = np.uint8(255 * x_show)
    x_show = cv2.applyColorMap(x_show, cv2.COLORMAP_JET)
    x_show = cv2.resize(x_show, (320, 320))
    img = cv2.resize(img, (320, 320))
    print(x_show.shape, img.shape)
    x_show = cv2.addWeighted(img, 0.5, x_show, 0.5, 0)

    if name is not None:
        cv2.imwrite('D:\pytorch\data\example\heatmap/' + name + '.jpg', x_show)
    cv2.imshow('img', x_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
