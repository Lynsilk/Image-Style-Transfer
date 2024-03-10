from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os,torch

class Image_Dataset(Dataset):
    def __init__(self, A_root, B_root, load_size, crop_size):
        self.transforms = transforms.Compose([
                transforms.Resize(load_size, interpolation = transforms.InterpolationMode.BICUBIC), #调整输入图片的大小，双三次插值法
                transforms.RandomCrop(crop_size), #随机裁剪
                transforms.RandomHorizontalFlip(),#随机水平翻转图像
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#归一化     
        ])
        self.A_paths = [os.path.join(A_root, f) for f in os.listdir(A_root)]
        self.B_paths = [os.path.join(B_root, f) for f in os.listdir(B_root)]

    def __getitem__(self, index):   #读取文件
        A_img = self.transforms(Image.open(self.A_paths[index % len(self.A_paths)]).convert('RGB'))
        B_img = self.transforms(Image.open(self.B_paths[index % len(self.B_paths)]).convert('RGB'))    
        return A_img, B_img

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

class Transfer:
    def __init__(self, checkpoint:str, device='cuda:0'):
        self.device = device
        self.model = torch.load(checkpoint)
        self.model.to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

    def transfer(self, image:str):
        image = Image.open(image).convert('RGB')
        width, height = image.size 
        image = transforms.Compose([
                transforms.Resize((height,width), interpolation = transforms.InterpolationMode.BICUBIC), #调整输入图片的大小，双三次插值法
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#归一化     
        ])(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        #图片保存，由张量转化为图片
        fake: torch.Tensor = self.model(image)
        fake = fake.data
        fake_numpy = fake[0].cpu().float().numpy()
        fake_numpy = (np.transpose(fake_numpy, (1, 2, 0)) + 1) / 2.0 * 255
        return fake_numpy.astype(np.uint8)
