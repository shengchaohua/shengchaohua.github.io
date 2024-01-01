#!/usr/bin/env sh

# 确保脚本抛出遇到的错误
set -e

# 生成静态文件
pnpm run docs:build

# 进入生成的文件夹
cd docs/.vuepress/dist

git init
git add -A
git commit -m 'deploy: pages'

# 发布到 https://<USERNAME>.github.io/<REPO>
git push -f --set-upstream https://github.com/shengchaohua/shengchaohua.github.io master:github-pages

cd -