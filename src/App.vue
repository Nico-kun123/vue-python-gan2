<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <header>
    <nav>
      <RouterLink to="/">Главная</RouterLink>
      <RouterLink to="/task">Задание</RouterLink>
    </nav>
  </header>

  <RouterView />
</template>

<style lang="scss" scoped>
header {
  line-height: 1.5;
  max-height: 100vh;
}
</style>
