<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const Images = ref([])

const generateImages = async () => {
  try {
    const response = await axios.get('http://127.0.0.1:5000/generate', {
      params: { images: 4 }
    })
    Images.value = response.data.images
    console.log(`✅ Были сгенерированы ${Images.value.length} примера изображений!`)
  } catch (error) {
    console.error('❌ Не получилось сгенерировать изображения:', error)
  }
}

onMounted(() => {
  generateImages()
})
</script>

<template>
  <div class="about">
    <div class="welcome">
      <h1>Задание</h1>
    </div>

    <p class="description">
      Создать веб-сайт, где будут отображаться сгенерированные изображения при нажатии на кнопку.
      <br /><br />
      В работе используется файл уже обученной модели генеративной нейросети (для GAN2). Обращение к
      серверу (Flask на Python), на котором работает модель, происходит с помощью axios.
      <br /><br />
      Сгенерированные изображения отправляются назад к клиенту и отображаются на странице. Клиент
      может ввести количество изображений, которые он хочет сгенерировать (максимум 100). Если поле
      пустое, то будет сгенерировано ОДНО изображение.
    </p>

    <div class="divider"></div>

    <div class="welcome">
      <h1>Примеры изображений:</h1>
    </div>

    <div class="images">
      <div v-for="(image, index) in Images" :key="index">
        <img :src="'data:image/png;base64,' + image" alt="image" width="128px" height="128px" />
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped></style>
