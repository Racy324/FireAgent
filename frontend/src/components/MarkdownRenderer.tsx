import ReactMarkdown from 'react-markdown'

interface Props {
  content: string
}

/** 渲染 Markdown 内容，适配暗色主题 */
export default function MarkdownRenderer({ content }: Props) {
  return (
    <ReactMarkdown
      className="prose prose-invert prose-sm max-w-none
        prose-headings:text-fire-300 prose-headings:font-bold
        prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2
        prose-p:text-gray-300 prose-p:leading-relaxed
        prose-strong:text-fire-300 prose-strong:font-semibold
        prose-li:text-gray-300 prose-li:marker:text-fire-500
        prose-a:text-fire-400 prose-a:no-underline hover:prose-a:underline
        prose-blockquote:border-fire-600 prose-blockquote:text-gray-400"
    >
      {content}
    </ReactMarkdown>
  )
}
