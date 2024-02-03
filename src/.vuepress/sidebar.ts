import {sidebar} from "vuepress-theme-hope";

export default sidebar({
  "/": [
    {
      text: "数据结构与算法",
      icon: "laptop-code",
      prefix: "da/",
      children: [
        {text: "数据结构", prefix: "data-structure/", children: "structure"},
        {text: "算法", prefix: "algorithm/", children: "structure"},
      ],
    },
    {
      text: "数据库",
      icon: "laptop-code",
      prefix: "database/",
      children: [
        {text: "数据结构", prefix: "data-structure/", children: "structure"},
        {text: "算法", prefix: "algorithm/", children: "structure"},
      ],
    },
    // {
    //   text: "文章",
    //   icon: "book",
    //   prefix: "posts/",
    //   children: "structure",
    // },
    // "intro",
    // {
    //   text: "幻灯片",
    //   icon: "person-chalkboard",
    //   link: "https://plugin-md-enhance.vuejs.press/zh/guide/content/revealjs/demo.html",
    // },
  ],
});
