import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI算法工程师数学地图',
  description: '面向AI算法工程师的数学知识体系、学习路线与算法应用说明',
  base: '/ai-math-map/',
  lang: 'zh-CN',

  // 开启数学公式渲染
  markdown: {
    math: true
  },

  // SEO与图标配置
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'keywords', content: 'AI算法,数学知识,线性代数,微积分,概率统计,深度学习' }]
  ],

  themeConfig: {
    siteTitle: 'AI Math Map',

    // 本地搜索（中文适配）
    search: {
      provider: 'local',
      options: {
        locales: {
          zh: {
            placeholder: '搜索数学知识点...',
            translations: {
              modal: {
                noResultsText: '未找到相关内容',
                resetButtonTitle: '清除搜索'
              }
            }
          }
        }
      }
    },

    // 顶部导航（补全下拉菜单）
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/guide/roadmap' },
      { text: '线性代数', link: '/linear-algebra/vector' },
      { text: '微积分', link: '/calculus/calculus-for-ml' },
      { text: '概率统计', link: '/probability/probability-distributions' },
      {
        text: '更多内容',
        items: [
          { text: '优化方法', link: '/optimization/optimization' },
          { text: '信息论', link: '/information-theory/information-theory' },
          { text: '进阶数学', link: '/advanced-math/complex-numbers-for-ai' }
        ]
      }
    ],

    // 侧边栏（支持折叠）
    sidebar: [
      {
        text: '学习路线',
        collapsible: true,
        collapsed: false,
        items: [
          { text: 'AI数学学习路线', link: '/guide/roadmap' }
        ]
      },
      {
        text: '线性代数',
        collapsible: true,
        collapsed: false,
        items: [
          { text: '线性代数直觉', link: '/linear-algebra/vector' },
          { text: '向量、矩阵与基本运算', link: '/linear-algebra/matrix' },
          { text: '矩阵变换', link: '/linear-algebra/matrix-transformations' },
          { text: '降维方法', link: '/linear-algebra/dimensionality-reduction' },
          { text: '奇异值分解 SVD', link: '/linear-algebra/svd' },
          { text: '张量操作', link: '/linear-algebra/tensor-operations' },
          { text: '范数与距离', link: '/linear-algebra/norms-distances' },
          { text: '线性方程组', link: '/linear-algebra/linear-systems' }
        ]
      },
      {
        text: '微积分',
        collapsible: true,
        collapsed: false,
        items: [
          { text: '机器学习中的微积分', link: '/calculus/calculus-for-ml' },
          { text: '链式法则与自动微分', link: '/calculus/chain-rule-autodiff' }
        ]
      },
      {
        text: '概率统计',
        collapsible: true,
        collapsed: false,
        items: [
          { text: '概率与分布', link: '/probability/probability-distributions' },
          { text: '贝叶斯公式', link: '/probability/bayes-theorem' },
          { text: '机器学习中的统计学', link: '/probability/statistics-for-ml' },
          { text: '采样方法', link: '/probability/sampling-methods' }
        ]
      },
      {
        text: '优化方法',
        collapsible: true,
        collapsed: true,
        items: [
          { text: '优化方法', link: '/optimization/optimization' },
          { text: '数值稳定性', link: '/optimization/numerical-stability' },
          { text: '凸优化', link: '/optimization/convex-optimization' }
        ]
      },
      {
        text: '信息论',
        collapsible: true,
        collapsed: true,
        items: [
          { text: '信息论', link: '/information-theory/information-theory' }
        ]
      },
      {
        text: '进阶数学',
        collapsible: true,
        collapsed: true,
        items: [
          { text: 'AI 中的复数', link: '/advanced-math/complex-numbers-for-ai' },
          { text: '傅里叶变换', link: '/advanced-math/fourier-transform' },
          { text: '机器学习中的图论', link: '/advanced-math/graph-theory-for-ml' },
          { text: '随机过程', link: '/advanced-math/stochastic-processes' }
        ]
      }
    ],

    // 右侧大纲
    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    // 翻页按钮文案
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新'
    },

    // GitHub社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/你的用户名/ai-math-map' }
    ],

    // 编辑此页链接
    editLink: {
      pattern: 'https://github.com/你的用户名/ai-math-map/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    // 页脚版权
    footer: {
      copyright: '© 2026 AI算法工程师数学地图 | 持续迭代更新'
    }
  }
})
