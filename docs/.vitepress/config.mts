import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI算法工程师数学地图',
  description: '面向AI算法工程师的数学知识体系、学习路线与算法应用说明',
  base: '/ai-math-map/',

  themeConfig: {
    siteTitle: 'AI Math Map',

    search: {
      provider: 'local'
    },

    nav: [
      { text: '学习路线', link: '/guide/roadmap' },
      { text: '线性代数', link: '/linear-algebra/matrix-multiplication' }
    ],

    sidebar: [
      {
        text: '学习路线',
        items: [
          { text: '总览', link: '/guide/roadmap' }
        ]
      },
      {
        text: '线性代数',
        items: [
          { text: '向量', link: '/linear-algebra/matrix-multiplication' }
        ]
      }
    ]
  }
})
