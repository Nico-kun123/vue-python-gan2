from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import torch
from torchvision.utils import save_image, make_grid
import os
import io
from PIL import Image
import base64
from generator import Generator

# Загрузка обученной модели
generator = Generator()
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
generator.eval()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Функция для конвертации изображений в base64
def convert_image_to_base64(img_tensor):
    buffer = io.BytesIO()
    save_image(img_tensor, buffer, format='PNG', normalize=True)
    buffer.seek(0)
    img = Image.open(buffer)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
    return base64_img

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Инициализация Flask
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def hello():
    return 'Это не простой сайт.<br/><br/>Пожалуйста, переходите по ссылке: "http://127.0.0.1:5000/generate?images=1".<br/><br/>Параметр "images" определяет количество генерируемых изображений (по умолчанию 1). '

@app.route('/generate', methods=['GET'])
def generate_image():
    # Получение параметра num_images из запроса (значение по умолчанию = 1)
    num_images = int(request.args.get('images', 1))
    
    # Генерация новых изображений
    with torch.no_grad():
        z = torch.randn(num_images, 100)
        generated_imgs = generator(z).cpu()
    
    # Конвертация сгенерированных изображений в base64
    images_base64 = [convert_image_to_base64(img) for img in generated_imgs]
    
    # Возвращение base64 изображений в ответе JSON
    return jsonify({'images': images_base64})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    app.run(debug=True)