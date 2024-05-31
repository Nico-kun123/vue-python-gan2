<script>
import { ref, onMounted, defineComponent } from 'vue'

import axios from 'axios'

export default defineComponent({
  name: 'HomeView',

  data() {
    return {
      generatedImages: []
    }
  },

  methods: {
    async handleSubmit(event) {
      event.preventDefault()

      const imgNumber = event.target.imgCount.value || 1

      // Небольшая валидация
      if (imgNumber < 1) {
        alert('ERROR: Количество изображений должно быть больше нуля!')
      } else if (imgNumber > 100) {
        alert('ERROR: Количество изображений не должно превышать 100!')
      } else {
        await this.generateImages(imgNumber)
      }

      // Очистка формы
      event.target.imgCount.value = ''
    },

    async generateImages(number_of_images) {
      axios
        .get('http://127.0.0.1:5000/generate', {
          params: { images: number_of_images }
        })
        .then((response) => {
          console.log(`✅ Были сгенерированы ${response.data.images.length} изображений!`)
          this.generatedImages = response.data.images
        })
        .catch((error) => {
          console.error('❌ Не получилось сгенерировать изображения:', error)
        })
    }
  }
})
</script>

<template>
  <main>
    <div class="welcome">
      <h1>Генерация (некачественных) изображений космоса</h1>
    </div>

    <p class="description">
      Введите количество изображений, которое вы хотите сгенерировать. Вы можете сгенерировать до
      100 изображений.
      <br /><br />
      Используйте кнопку "Генерировать".
    </p>

    <form @submit="handleSubmit" class="form">
      <input type="number" placeholder="Количество изображений" id="imgCount" name="imgCount" />
      <button type="submit">Генерировать</button>
    </form>

    <div class="divider"></div>

    <div class="images">
      <div v-for="(image, index) in generatedImages" :key="index">
        <img :src="'data:image/png;base64,' + image" alt="image" width="128px" height="128px" />
      </div>
    </div>
  </main>
</template>

<style lang="scss" scoped>
form {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20px;

  width: 100%;

  input {
    margin: 10px 0;
    border-radius: 5px;
    padding: 5px;
    width: 50%;
    min-width: 100px;
    max-width: 500px;
  }
}
</style>
