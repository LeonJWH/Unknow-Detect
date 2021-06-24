import numpy as np
import cv2, torch
from torchvision.models import resnet50
import torchvision.transforms as tf

class CLS():
    def __init__(self):
        self.model = resnet50(pretrained=True).eval()
        self.output = torch.nn.Softmax(dim=1)

    def prapare_img(self, im):
        w, h, c = im.shape[1], im.shape[0], im.shape[2]

        new_im = np.zeros((224, 224, 3),dtype=np.float32)
        if w >= h:
            new_h = int(h * 224 / w)
            _im = cv2.resize(im, (224, new_h))
            new_im[(112 - int(new_h / 2)):(112 - int(new_h / 2) + new_h), :, :] = _im

        else:
            new_w = int(w * 224 / h)
            _im = cv2.resize(im, (new_w, 224))
            new_im[:, (112 - int(new_w / 2)):(112 - int(new_w / 2) + new_w), :] = _im

        return new_im

    def predict(self, img):
        # image = cv2.resize(img, (224, 224))
        image = self.prapare_img(img)
        image = np.float32(image) / 255.0
        image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        image = image.transpose((2, 0, 1))
        input_x = torch.from_numpy(image).unsqueeze(0)

        pred = self.model(input_x)
        pred = self.output(pred).squeeze().detach().numpy()

        return pred
