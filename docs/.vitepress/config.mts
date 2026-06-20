import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI算法工程师数学地图',
  description: '面向AI算法工程师的数学知识体系、学习路线与算法应用说明',

  base: '/ai-math-map/',

  lastUpdated: true,

  themeConfig: {
    siteTitle: 'AI Math Map',

    search: {
      provider: 'local'
    },

    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/guide/roadmap' },
      { text: '线性代数', link: '/linear-algebra/vector' }
    ],

    sidebar: [
      {
        text: '学习路线',
        items: [
          { text: 'AI数学学习路线', link: '/guide/roadmap' }
        ]
      },
      {
        text: '线性代数',
        items: [
          { text: '线性代数直觉', link: '/linear-algebra/vector' },
          { text: '向量、矩阵与基本运算', link: '/linear-algebra/matrix' },
          { text: '矩阵变换', link: '/linear-algebra/matrix-transformations' }
        ]
      }
    ],

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新'
    }
  }
})
