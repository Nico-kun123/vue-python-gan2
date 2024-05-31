# САЙТ ДЛЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ С ПОМОЩЬЮ НЕЙРОСЕТИ GAN2 (Vue.js + Python,Flask)

> **Примечание:** Этот проект НЕ требует постоянных обновлений кода.

> **Статус Проекта:** 🟩 Актуален.

## 📑Содержание

1. [Цели и Задачи](#-цели-и-задачи)
2. [Основная работа](#основная-работа)

   a) [Главная страница](#главная-страница)
   
   b) [Страница с небольшим описанием задания](#страница-с-небольшим-описанием-задания)
   
   c) [Генеративно-состязательная нейросеть (GAN2)](#генеративно-состязательная-нейросеть-gan2)

   - [Немного про саму нейросеть](#немного-про-саму-нейросеть)
   - [Как я обучал свою нейросеть](#как-я-обучал-свою-нейросеть)

4. [Стек технологий](#-стек-технологий)
5. [Установка](#-установка)

---

# ❗ Цели и задачи

Целью данной работы является реализация взаимодействия с моделью генеративно-состязательной нейросети (GAN2) через веб-сайт. Данная модель (её Generator) уже предварительно обучена.

---

# Основная работа

Работа состоит из двух простых страниц. Подробнее о них ниже.

## Главная страница

На главной странице есть форма, состоящая из одного поля и кнопки submit. Пользователь вводит целое число — количество изображений, которое требуется сгенерировать. Если отправить пустую форму, то сгенерируется 1 изображение.

Для этого поля есть следующие ограничения:

- число изображений не может быть меньше 1;
- число изображений не может быть больше 100.

Результат генерации изображений:

![image](https://github.com/Nico-kun123/vue-python-gan2/assets/77405288/0d0c6dda-3ed5-404f-960a-a69f04292c1b)

## Страница с небольшим описанием задания

Здесь находится небольшая информация о том, что из себя представляет эта работа.

Внешний вид страницы:

![image](https://github.com/Nico-kun123/vue-python-gan2/assets/77405288/8b6743e9-d0d3-4f21-b821-23ff0ca8d9ff)

## Генеративно-состязательная нейросеть (GAN2)

### Немного про саму нейросеть

_Генеративно-состязательная нейросеть (Generative adversarial network, GAN)_ — архитектура, состоящая из генератора и дискриминатора, настроенных на работу друг против друга. Отсюда GAN и получила название генеративно-созтязательная.

Как работает генеративно-состязательная нейросеть?

- Одна нейронная сеть, называемая генератором, генерирует новые экземпляры данных, а другая — дискриминатор, оценивает их на подлинность; т.е. дискриминатор решает, относится ли каждый экземпляр данных, который он рассматривает, к набору тренировочных данных или нет.

**Дискриминатор**: Дискриминационные алгоритмы пытаются классифицировать входные данные. Учитывая особенности полученных данных, они стараются определить категорию, к которой они относятся.

**Генератор**: Генеративные алгоритмы заняты обратным. Вместо того, чтобы предсказывать категорию по имеющимся образам, они пытаются подобрать образы к данной категории.

Шаги, которые проходит GAN:

1. Генератор получает рандомное число и возвращает изображение.
2. Это сгенерированное изображение подается в дискриминатор наряду с потоком изображений, взятых из фактического набора данных.
3. Дискриминатор принимает как реальные, так и поддельные изображения и возвращает вероятности, числа от 0 до 1, причем 1 представляет собой подлинное изображение и 0 представляет фальшивое.

🤢 **GAN требуют много времени на тренировку**. На одном GPU тренировка может занимать часы, а на одном CPU — более одного дня.

### Как я обучал свою нейросеть

> **Примечание**: данная модель не является достаточно хорошей. На данный момент нет более хорошей альтернативы.

Генерация изображений происходит с помощью файла "generator.pth", который представляет собой обученную модель генеративно-состязательной нейросети (GAN2) — это нейросеть для генератора. Скачать этот файл можно здесь: <https://drive.google.com/drive/folders/1Uuw-Kl-AzuB2i6lJpQUnIBrjapzghwiY?usp=sharing>

Архив с данными для обучения я скачал с Kaggle (<https://www.kaggle.com/datasets/kimbosoek/cosmos-images>). Его нужно скачать себе на компьютер. Также этот архив можно скачать, используя ссылку на Google Drive выше.

❗ **Вот код, который я использовал для обучения GAN2 (сделано в Google Colab)**:

Для Colab:

```python
!pip install torch torchvision numpy matplotlib scipy pytorch-fid scipy
```

Импорты:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F
from google.colab import files
import zipfile
import os
from PIL import Image
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
```

Загрузка архива (НУЖНО ВЫБРАТЬ СКАЧЕННЫЙ ФАЙЛ НА КОМПЬЮТЕРЕ):

```python
uploaded = files.upload()
archive_name = list(uploaded.keys())[0]
with zipfile.ZipFile(archive_name, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')
print(os.listdir('/content/dataset'))
```

Подготовка dataset и dataloader:

```python
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
dataset = CustomDataset(root_dir='/content/dataset/data/img_align_celeba', transform=transform)
filtered_dataset = [data for data in dataset if data is not None]
dataloader = DataLoader(filtered_dataset, batch_size=32, shuffle=True)
```

Архитектура моделей для GAN2 (и другие важные для дальнейшего обучения вещи):

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128*128*3),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 3, 128, 128)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128*128*3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(-1, 128*128*3))

# Создание директории для хранения сгенерированных изображений
os.makedirs('/content/generated_images', exist_ok=True)
# Создание директории для хранения losses
os.makedirs('/content/G_D_losses', exist_ok=True)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Learning rate schedulers
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.1)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=30, gamma=0.1)
```

Функция для сохранения графиков изменения loss:

```python
def save_losses(g_losses, d_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"/content/G_D_losses/losses_epoch_{epoch}.png")
    plt.close()
```

Обучение генератора и дискриминатора:

```python
g_losses = []
d_losses = []

DL_size = len(dataloader);

num_epochs = 50
latent_dim = 100

for epoch in range(num_epochs):
    for i, imgs in enumerate(dataloader):
        batch_size = imgs.size(0)

        real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Label smoothing для реальных данных
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # Label smoothing для фейковых данных

        real_imgs = imgs.to(device)

        # Train discriminator
        optimizer_d.zero_grad()

        z = torch.randn(batch_size, latent_dim).to(device)
        generated_imgs = generator(z)

        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(generated_imgs.detach()), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()

        fake_labels.fill_(0.9)  # Fake labels are real for generator cost
        g_loss = criterion(discriminator(generated_imgs), fake_labels)
        g_loss.backward()
        optimizer_g.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if i == 0 or i % (DL_size-1) == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)-1} \
                  Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # Step learning rate schedulers
    scheduler_g.step()
    scheduler_d.step()

    save_image(generated_imgs.data[:25], f"/content/generated_images/epoch_{epoch}.png", nrow=5, normalize=True)
    save_losses(g_losses, d_losses, epoch)
```

Генерация изображений (проверка обученной модели генератора):

```python
num_images = 500
latent_dim = 100

desired_height = 128
desired_width = 128
resize_transform = transforms.Resize((desired_width, desired_height))

generator.eval()
os.makedirs('/content/generated_images_testing', exist_ok=True)
with torch.no_grad():
    for i in range(num_images // 32):  # Пакеты по 32 изображения
        z = torch.randn(32, latent_dim).to(device)
        generated_imgs = generator(z).cpu()
        for j, img in enumerate(generated_imgs):
            resized_img = resize_transform(img)
            save_image(resized_img, f'/content/generated_images_testing/img_{i*32 + j}.png', normalize=True)
```

Вычисление метрики "Frechet Inception Distance" (FID) для оценки качества сгенерированных изображений (**НЕ РАБОТАЕТ!!!**):

```python
# Пути к директориям с реальными и сгенерированными изображениями
path_real = '/content/dataset/data/img_align_celeba'
path_generated = '/content/generated_images_testing'

# Проверка и удаление NaN и бесконечных значений
def clean_data(path_real):
    for fname in os.listdir(path_real):
        try:
            img = Image.open(os.path.join(path_real, fname))
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f'Bad file {fname}: {e}')
            os.remove(os.path.join(path_real, fname))

clean_data(path_real)
clean_data(path_generated)

# Рассчет FID
fid_value = fid_score.calculate_fid_given_paths([path_real, path_generated], batch_size=32, device='cuda', dims=2048)
print(f'FID: {fid_value}')
```

---

# 💻 Стек технологий

В проекте был использован следующий стек технологий:

- [HTML](https://developer.mozilla.org/ru/docs/Learn/HTML/Introduction_to_HTML)
- [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS)
- [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
- [Vue.js](https://vuejs.org)
- [Git](https://git-scm.com/)
- [Sass](https://sass-lang.com/)
- [Vite.js](https://vitejs.dev)
- [Python](https://www.python.org)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)

---

# ⏬ Установка

Клонируем удалённый репозиторий на локальную машину:

```markdown
git clone https://github.com/Nico-kun123/vue-python-gan2
```

Устанавливаем все необходимые компоненты, создаём виртуальную среду для Python:

```markdown
cd vue-python-gan2
npm install
python3 -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

Затем нужно запустить Flask-приложение с моделью нейросети:

```markdown
npm run startServer
```

Потом нужно запустить сайт:

```markdown
npm run dev
```

После этого нужно открыть страницу <http://localhost:5173/> в браузере.

Содержание <code>package.json</code>:

```json
{
  "name": "vue-python-ai",
  "version": "0.0.0",
  "private": true,
  "author": {
    "name": "Nicolay Kudryavtsev"
  },
  "type": "module",
  "scripts": {
    "dev": "vite",
    "startServer": "python app.py",
    "format": "prettier --write src/",
    "build": "vite build",
    "preview": "npm run build && vite preview"
  },
  "dependencies": {
    "vue": "^3.4.21",
    "vue-router": "^4.3.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.4",
    "axios": "^1.7.2",
    "prettier": "^3.2.5",
    "sass": "^1.77.4",
    "vite": "^5.2.8"
  }
}
```

В проекте есть следующие разделы:

- **dependencies**: Это пакеты, необходимые для работы приложения.
- **devDependencies**: Это пакеты, которые нужны только для разработки и тестирования приложения. Они не будут включены в окончательную сборку приложения.

В проекте есть следующие скрипты (только первые 3 важны):

- **"dev"**. Этот скрипт запускает сервер разработки Vite на локальной машине (<http://localhost:5173/>);
- **"startServer"**. Этот скрипт запускает приложение Flask на локальной машине (<http://127.0.0.1:5000>);
- **"format"**. Автоматически форматирует код, используя "Prettier";
- **"build"**. Этот скрипт используется для сборки проекта для production. Он минимизирует и оптимизирует код для лучшей производительности в production;
- **"preview"**. Этот скрипт предназначен для предварительного просмотра собранного проекта. Он запускает сервер, который позволяет увидеть, как он будет выглядеть и работать в production.

---

## Автор

Кудрявцев Николай (Электронная почта: <nicolay.kudryavtsev@gmail.com>)
