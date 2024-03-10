import torch
import random

#图片缓冲池
class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.imgs = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.imgs.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    index = random.randint(0, self.pool_size-1)
                    tmp = self.imgs[index].clone()
                    self.imgs[index] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
