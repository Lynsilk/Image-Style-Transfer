from utils.image_dataset import Transfer
from PIL import Image
import torch

if __name__ == '__main__':
    # 导入模型，进行风格迁移.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transfer('checkpoints/sketch/latest_netG_B.pth', device)
    output = model.transfer('../data/anime/testB/0000.jpg')
    image = Image.fromarray(output)
    image.show()
