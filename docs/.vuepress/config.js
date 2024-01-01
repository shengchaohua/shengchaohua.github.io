import { defineUserConfig } from 'vuepress'
import { defaultTheme } from 'vuepress'

export default defineUserConfig({
    // set site base to default value
    base: '/',

    // site-level locales config
    locales: {
        '/': {
            lang: 'zh-CN',
            title: '绳子的学习笔记',
            description: '绳子的学习笔记',
        },
    },

    lang: 'zh-CN',
    title: '绳子的学习笔记',
    description: '绳子的学习笔记',
    theme: defaultTheme({
        docsDir: 'docs',

        navbar: [
            {
                text: '首页',
                link: '/'
            },
            {
                text: "数据结构与算法",
                link: "/da/"
            },
        ],

        // 侧边栏对象
        // 不同子路径下的页面会使用不同的侧边栏
        sidebar: {
            '/da/': [
                {
                    text: '数据结构',
                    collapsible: true,
                    children: [
                        '/da/ds/array.md',
                        '/da/ds/linklist.md',
                    ],
                },
                {
                    text: '算法',
                    collapsible: true,
                    children: [
                        '/da/alg/README.md',
                    ],
                },
            ],
        },
        sidebarDepth: 0,

        contributors: false,
        lastUpdatedText: "上次更新时间",
    }),
})