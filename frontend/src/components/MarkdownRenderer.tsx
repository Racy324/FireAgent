import ReactMarkdown from 'react-markdown'
import type { ReactNode } from 'react'

interface Props {
  content: string
}

/** 渲染 Markdown 内容，适配暗色主题 */
export default function MarkdownRenderer({ content }: Props) {
  const normalizedContent = normalizeAnswerMarkdown(stripSourceSection(content))

  return (
    <ReactMarkdown
      className="prose prose-invert prose-sm max-w-none
        prose-headings:text-fire-300 prose-headings:font-bold
        prose-h3:text-sm prose-h3:mt-4 prose-h3:mb-2 prose-h3:tracking-normal
        prose-p:text-gray-300 prose-p:leading-7 prose-p:my-2
        prose-strong:text-fire-300 prose-strong:font-semibold
        prose-li:text-gray-300 prose-li:marker:text-fire-500 prose-li:leading-7
        prose-a:text-fire-400 prose-a:no-underline hover:prose-a:underline
        prose-blockquote:border-fire-600 prose-blockquote:text-gray-400"
      components={{
        h3: ({ children }) => (
          <h3 className="mt-4 mb-2 inline-flex items-center rounded-md border border-fire-500/20 bg-fire-500/10 px-2.5 py-1 text-sm font-semibold text-fire-200">
            {children}
          </h3>
        ),
        p: ({ children }) => (
          <p className="my-2 leading-7 text-gray-300">
            {renderCitationBadges(children)}
          </p>
        ),
        li: ({ children }) => (
          <li className="leading-7 text-gray-300">
            {renderCitationBadges(children)}
          </li>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-fire-200">
            {renderCitationBadges(children)}
          </strong>
        ),
      }}
    >
      {normalizedContent}
    </ReactMarkdown>
  )
}

const SECTION_TITLES = [
  '直接回答',
  '依据说明',
  '补充分析',
  '不确定性',
]

function stripSourceSection(content: string): string {
  return content.replace(
    /(^|\n)(?:#+\s*)?资料来源\s*[:：]?\s*\n[\s\S]*?(?=\n(?:#+\s*)?(?:直接回答|依据说明|补充分析|不确定性)\s*[:：]?\s*(?:\n|$)|$)/g,
    '$1',
  ).trim()
}

function normalizeAnswerMarkdown(content: string): string {
  const titlePattern = SECTION_TITLES.join('|')
  return content.replace(
    new RegExp(`(^|\\n)(${titlePattern})\\s*[:：]?\\s*(?=\\n|$)`, 'g'),
    '$1### $2',
  )
}

function renderCitationBadges(children: ReactNode): ReactNode {
  if (typeof children === 'string') {
    return splitCitationText(children)
  }
  if (Array.isArray(children)) {
    return children.map((child, index) => (
      <span key={index}>
        {renderCitationBadges(child)}
      </span>
    ))
  }
  return children
}

function splitCitationText(text: string): ReactNode {
  const parts = text.split(/(\[(?:L|W)\d+\])/g)
  if (parts.length === 1) return text

  return parts.map((part, index) => {
    if (/^\[(?:L|W)\d+\]$/.test(part)) {
      return (
        <span
          key={`${part}-${index}`}
          className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded border border-fire-500/25 bg-fire-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold leading-none text-fire-300"
        >
          {part}
        </span>
      )
    }
    return part
  })
}
