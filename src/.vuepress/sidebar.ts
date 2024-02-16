import {sidebar} from "vuepress-theme-hope";

export default sidebar({
  "/da/": "structure",
  "/pl/": "structure",
  "/database/": "structure",
  // fallback
  "/": [
    "" /* / */,
    "intro" /* /intro.html */,
  ],
});
