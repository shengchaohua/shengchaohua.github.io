import { defineUserConfig } from "vuepress";
import theme from "./theme.js";

export default defineUserConfig({
  base: "/shengchaohua.github.io/",

  lang: "zh-CN",
  title: "Max",
  description: "",

  theme,

  // 和 PWA 一起启用
  // shouldPrefetch: false,
});
