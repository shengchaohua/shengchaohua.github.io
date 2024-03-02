import {navbar} from "vuepress-theme-hope";

export default navbar([
  "/",
  {
    text: "计算机基础",
    icon: "pen-to-square",
    children: [
      {text: "数据结构与算法", link: "computer-basic/da/"},
      {text: "计算机网络", link: "computer-basic/computer-network/"},
      {text: "操作系统", link: "computer-basic/os/"},
    ]
  },
  {
    text: "编程语言",
    icon: "pen-to-square",
    children: [
      {text: "Go", link: "pl/go/"},
      {text: "Java", link: "pl/java/"},
    ]
  },
  {
    text: "数据库",
    icon: "pen-to-square",
    children: [
      {text: "MySQL", link: "database/mysql/"},
      {text: "Redis", link: "database/redis/"},
    ]
  },
  {
    text: "网站相关",
    icon: "pen-to-square",
    link: "intro.md",
  }
]);
