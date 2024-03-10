import re  
import matplotlib.pyplot as plt  

name='sketch'  
# 日志文件路径  
log_file_path = 'checkpoints/'+name+'/log'  
  
# 用于存储数据的列表
if log_file_path == 'checkpoints/cartoon/log':
    data_num=31
if log_file_path == 'checkpoints/ink/log':
    data_num=11
if log_file_path == 'checkpoints/sketch/log':
    data_num=3
epochs = []  
D_A = []  
D_B = []  
GAN_A = []
GAN_B = []
Cycle_A = []
Cycle_B = []
idt_A = []
idt_B = []
  
# 正则表达式模式，用于匹配数字和浮点数  
pattern = r'[-+]?\d*\.\d+|\d+'  
  
# 读取日志文件，并每隔一行读取数据  
with open(log_file_path, 'r') as log_file:  
    # 跳过表头行  
    next(log_file)  
    for line_no, line in enumerate(log_file):  
        # 如果行号不是偶数，则跳过这一行（从0开始计数，所以实际跳过奇数行）  
        if line_no % data_num != 0:  
            continue  
          
        # 使用正则表达式查找所有的数字和浮点数  
        matches = re.findall(pattern, line)  
        # 假设匹配项的顺序与列的顺序一致  
        epoch, _, da, db, ga, gb, ca, cb, ida, idb, *rest = map(float, matches)  
          
        epochs.append(epoch)  
        D_A.append(da)  
        D_B.append(db)  
        GAN_A.append(ga)  
        GAN_B.append(gb)  
        Cycle_A.append(ca)  
        Cycle_B.append(cb)  
        idt_A.append(ida)  
        idt_B.append(idb)  

def image_paint(data_A,label_A,title_A,data_B,label_B,title_B,x_label,y_label):
    # 创建一个新的图形窗口  
    plt.figure(figsize=(6, 5))  
    # 绘制D_A随EPOCH变化的折线图，作为第一个子图  
    plt.subplot(2, 1, 1)  # 2行1列的子图，当前是第1个子图  
    plt.plot(epochs, data_A, label=label_A,color=(0.12, 0.47, 0.70))  
    plt.title(title_A,color=(0.12, 0.47, 0.70))  
    plt.xlabel(x_label)  
    plt.ylabel(y_label)  
    plt.grid(True)  
    plt.legend()  
    # 绘制D_B随EPOCH变化的折线图，作为第二个子图  
    plt.subplot(2, 1, 2)  # 2行1列的子图，当前是第2个子图  
    plt.plot(epochs, data_B, label=label_B,color='green')  
    plt.title(title_B,color='green')  
    plt.xlabel(x_label)  
    plt.ylabel(y_label)  
    plt.grid(True)  
    plt.legend()  
    # 调整子图之间的间距  
    plt.tight_layout()  
    # 显示图形  
    plt.show()

if __name__ == '__main__':  
    image_paint(D_A,'D_A',name+' : D_A LOSS',D_B,'D_B',name+' : D_B LOSS','epoch','loss')
    image_paint(GAN_A,'GAN_A',name+' : GAN_A LOSS',GAN_B,'GAN_B',name+' : GAN_B LOSS','epoch','loss')
    image_paint(Cycle_A,'Cycle_A',name+' : Cycle_A LOSS',Cycle_B,'Cycle_B',name+' : Cycle_B LOSS','epoch','loss')
    image_paint(idt_A,'idt_A',name+' : idt_A LOSS',idt_B,'idt_B',name+' : idt_B LOSS','epoch','loss')