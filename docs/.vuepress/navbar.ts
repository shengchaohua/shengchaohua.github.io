import {navbar} from "vuepress-theme-hope";

export default navbar([
  "/",
  {
    text: "计算机基础",
    icon: "pen-to-square",
    children: [
      {text: "数据结构与算法", link: "computer-basic/da/"},
    ]
  },
]);
