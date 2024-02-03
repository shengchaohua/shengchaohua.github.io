import { defineUserConfig } from "vuepress";
import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "绳子的学习笔记",
  description: "绳子的学习笔记",

  theme,

  // 和 PWA 一起启用
  // shouldPrefetch: false,
});
