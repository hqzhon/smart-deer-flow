export function autoFixMarkdown(markdown: string): string {
  return autoCloseTrailingLink(markdown);
}

function autoCloseTrailingLink(markdown: string): string {
  // Fix unclosed Markdown links or images, but preserve code blocks
  let fixedMarkdown: string = markdown;

  // Split content by code blocks to avoid processing content inside them
  const codeBlockRegex = /```[\s\S]*?```/g;
  const codeBlocks: string[] = [];
  let codeBlockIndex = 0;
  
  // Replace code blocks with placeholders
  fixedMarkdown = fixedMarkdown.replace(codeBlockRegex, (match) => {
    const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
    codeBlocks[codeBlockIndex] = match;
    codeBlockIndex++;
    return placeholder;
  });

  // Fix unclosed image syntax ![...](...)
  fixedMarkdown = fixedMarkdown.replace(
    /!\[([^\]]*)\]\(([^)]*)$/g,
    (match: string, altText: string, url: string): string => {
      return `![${altText}](${url})`;
    },
  );

  // Fix unclosed link syntax [...](...)
  fixedMarkdown = fixedMarkdown.replace(
    /\[([^\]]*)\]\(([^)]*)$/g,
    (match: string, linkText: string, url: string): string => {
      return `[${linkText}](${url})`;
    },
  );

  // Fix unclosed image syntax ![...]
  fixedMarkdown = fixedMarkdown.replace(
    /!\[([^\]]*)$/g,
    (match: string, altText: string): string => {
      return `![${altText}]`;
    },
  );

  // Fix unclosed link syntax [...] - but only for actual Markdown links, not Mermaid nodes
  // Only fix if it's at the very end of the content and not part of a Mermaid diagram
  fixedMarkdown = fixedMarkdown.replace(
    /\[([^\]]*)$/g,
    (match: string, linkText: string): string => {
      // Don't fix if this appears to be part of a Mermaid diagram
      // Check if the line contains Mermaid syntax like "-->" or "--->"
      const lines = fixedMarkdown.split('\n');
      const lastLine = lines[lines.length - 1];
      if (lastLine && (lastLine.includes('-->') || lastLine.includes('--->'))) {
        return match; // Don't modify Mermaid syntax
      }
      return `[${linkText}]`;
    },
  );

  // Fix unclosed images or links missing ")"
  fixedMarkdown = fixedMarkdown.replace(
    /!\[([^\]]*)\]\(([^)]*)$/g,
    (match: string, altText: string, url: string): string => {
      return `![${altText}](${url})`;
    },
  );

  fixedMarkdown = fixedMarkdown.replace(
    /\[([^\]]*)\]\(([^)]*)$/g,
    (match: string, linkText: string, url: string): string => {
      return `[${linkText}](${url})`;
    },
  );

  // Restore code blocks
  for (let i = 0; i < codeBlocks.length; i++) {
    const codeBlock = codeBlocks[i];
    if (codeBlock) {
      fixedMarkdown = fixedMarkdown.replace(`__CODE_BLOCK_${i}__`, codeBlock);
    }
  }

  return fixedMarkdown;
}
