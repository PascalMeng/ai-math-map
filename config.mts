import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI算法工程师数学地图',
  description: '面向AI算法工程师的数学知识体系、学习路线与算法应用说明',

  // 作用：设置 GitHub Pages 子路径。
  // 为什么：如果你的仓库名是 ai-math-map，部署地址通常是：
  // https://你的用户名.github.io/ai-math-map/
  // 所以 base 必须写成 /ai-math-map/，否则 CSS、JS 路径可能加载失败。
  base: '/ai-math-map/',

  lastUpdated: true,

  themeConfig: {
    siteTitle: 'AI Math Map',

    search: {
      provider: 'local'
    },

    nav: [
      { text: '学习路线', link: '/guide/roadmap' },
      { text: '线性代数', link: '/linear-algebra/vector' },
      { text: '微积分与优化', link: '/calculus-optimization/derivative' },
      { text: '概率统计', link: '/probability-statistics/random-variable' }
    ],

    sidebar: [
      {
        text: '学习路线',
        items: [
          { text: '总览', link: '/guide/roadmap' },
          { text: '入门路线', link: '/guide/beginner' },
          { text: '论文阅读路线', link: '/guide/paper-reading' }
        ]
      },
      {
        text: '线性代数',
        items: [
          { text: '向量', link: '/linear-algebra/vector' },
          { text: '矩阵', link: '/linear-algebra/matrix' },
          { text: '矩阵乘法', link: '/linear-algebra/matrix-multiplication' }
        ]
      },
      {
        text: '微积分与优化',
        items: [
          { text: '导数', link: '/calculus-optimization/derivative' },
          { text: '梯度', link: '/calculus-optimization/gradient' },
          { text: '梯度下降', link: '/calculus-optimization/gradient-descent' }
        ]
      },
      {
        text: '概率论与统计',
        items: [
          { text: '随机变量', link: '/probability-statistics/random-variable' },
          { text: '概率分布', link: '/probability-statistics/distribution' },
          { text: '最大似然估计', link: '/probability-statistics/mle' }
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