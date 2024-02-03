import {navbar} from "vuepress-theme-hope";

export default navbar([
  "/",
  {
    text: "计算机基础",
    link: "",
    children: [
      {
        text: "数据结构与算法",
        icon: "pen-to-square",
        prefix: "da/",
        children: [
          {text: "数据结构", icon: "pen-to-square", link: "data-structure"},
          {text: "算法", icon: "pen-to-square", link: "algorithm"},
        ],
      },
    ]
  },
]);
