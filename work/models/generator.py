import torch
from torch import nn

# in_channels ：输入层通道数
# mid_channels：中间层通道数
# out_channels：输出层通道数
# num_block   ：层块数【残差块】

def build_G(in_channels, mid_channels, out_channels, num_block):
    net = ResNet(in_channels, mid_channels, out_channels, num_block=num_block)
    return net

#构造生成器残差神经网络模型
class ResNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_block):
        super(ResNet, self).__init__()#调用父类构造函数
        network=[]
        #————————————————————【编码器：卷积层*3】————————————————————
        encoder = [#—————————————————conv_1————————————————————
                nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(True),
                #————————————————————conv_2————————————————————
                nn.Conv2d(mid_channels * 1, mid_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(mid_channels * 2),
                nn.ReLU(True),
                #————————————————————conv_3————————————————————
                nn.Conv2d(mid_channels * 2, mid_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(mid_channels * 4),
                nn.ReLU(True),
        ]
        network+=encoder
        #————————————————————【转换器：残差块*num_block】————————————————————
        for i in range(num_block):
            network += [ResBlock(mid_channels * 4, mid_channels * 4)]
        #————————————————————【解码器：反卷积*3】————————————————————
        decoder = [#—————————————————conv_1————————————————————
                nn.ConvTranspose2d(mid_channels * 4, mid_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(mid_channels * 2),
                nn.ReLU(True),
                #————————————————————conv_2————————————————————
                nn.ConvTranspose2d(mid_channels * 2, mid_channels * 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(mid_channels * 1),
                nn.ReLU(True),
                #————————————————————conv_3————————————————————
                nn.Conv2d(mid_channels, out_channels, kernel_size=7, stride=1, padding=3,padding_mode='reflect'),
                nn.Tanh(),
        ]
        network+=decoder
        self.model = nn.Sequential(*network)

    def forward(self, x):
        return self.model(x)

#构造残差块
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        block=[
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.InstanceNorm2d(out_channel),
        ]
        self.conv_block = nn.Sequential(*block)
    
    def forward(self, x):
        return x + self.conv_block(x)

if __name__ == '__main__':
    model = ResNet(3, 64, 3, 9)
    print(model)

    x = torch.ones(size=(1, 3, 256, 256))
    y = model(x)
    print(y.shape)