import {sidebar} from "vuepress-theme-hope";

export default sidebar({
  "/computer-basic/": "structure",
  "/pl/": "structure",
  "/database/": "structure",
  "/mq/": "structure",
  // fallback
  "/": [
    "" /* / */,
    "intro" /* /intro.html */,
  ],
});
