from flask import Flask, render_template, request, jsonify  
from utils.image_dataset import Transfer
from PIL import Image
import torch,io,base64,webbrowser,threading

WebGUI = Flask(__name__)  
WebGUI.static_folder = 'web/static'             #设置静态文件夹路径 
WebGUI.template_folder='web/templates'          #设置模板文件夹路径
WebGUI.config['UPLOAD_FOLDER'] = 'web/uploads'  #设置上传文件夹路径

#定义路由和视图函数 
@WebGUI.route('/')  
def index():  
    return render_template('index.html')  
  
@WebGUI.route('/transfer', methods=['POST'])  
def process_image():  
    #获取上传图片文件
    file = request.files['image']
    #获取风格列表选项  
    style_list = request.form.get('style_list')  
    #加载对应风格模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if style_list == 'ink':  
        model = Transfer('checkpoints/ink/latest_netG_B.pth', device)
    if style_list == 'sketch':  
        model = Transfer('checkpoints/sketch/latest_netG_B.pth', device)
    if style_list == 'cartoon':  
        model = Transfer('checkpoints/cartoon/70_netG_B.pth', device)
    #执行风格迁移操作
    output = model.transfer(file)
    image = Image.fromarray(output)   
    #将图片转换为Base64编码  
    buffered = io.BytesIO()  
    image.save(buffered, format="PNG")  
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  
    return jsonify({'image_data': img_str})  
  
if __name__ == '__main__':  
    threading.Timer(2.0, lambda: webbrowser.open('http://127.0.0.1:5000/')).start() 
    WebGUI.run(debug=False)

