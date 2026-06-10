/** 格式化年份分布数据为 ECharts 柱状图数据 */
export function formatYearDistribution(dist: Record<string, number>) {
  const sorted = Object.entries(dist)
    .map(([year, count]) => ({ year: Number(year), count }))
    .filter((d) => d.year > 2000 && d.year < 2100)
    .sort((a, b) => a.year - b.year)
  return sorted
}

/** 格式化关键词频率为 ECharts 词云数据 */
export function formatKeywordCloud(keywords: Record<string, number>) {
  const result = Object.entries(keywords)
    .filter(([name]) => name.length >= 2)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 25)

  // 如果关键词为空，返回空数组（前端会显示空状态）
  return result
}

/** 从论文标题中提取高频词作为词云兜底数据 */
export function extractKeywordsFromTitles(papers: { title: string }[]): { name: string; value: number }[] {
  const counter: Record<string, number> = {}
  const stopWords = new Set(['基于', '的', '研究', '与', '及', '在', '中', '对', '和', '方法', '技术', '应用', '分析', '实现', '设计', '优化', '系统', '算法', '模型', '检测', '识别'])

  for (const paper of papers) {
    // 提取 2-6 字的中文词组
    const matches = paper.title.match(/[一-龥]{2,6}/g) || []
    for (const word of matches) {
      if (!stopWords.has(word) && word.length >= 2) {
        counter[word] = (counter[word] || 0) + 1
      }
    }
  }

  return Object.entries(counter)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 25)
}

/** 截断文本 */
export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text
  return text.slice(0, maxLen) + '...'
}

/** 格式化耗时 */
export function formatElapsed(seconds?: number): string {
  if (seconds === undefined) return ''
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  return `${seconds.toFixed(1)}s`
}
