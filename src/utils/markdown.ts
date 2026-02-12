// Utility functions for processing markdown content
export const extractHeadings = (markdown: string) => {
  const headingRegex = /^(\#{1,6})\s+(.*)$/gm;
  const headings = [];
  let match;

  while ((match = headingRegex.exec(markdown)) !== null) {
    const level = match[1].length;
    const title = match[2];
    headings.push({ level, title });
  }

  return headings;
};

export const extractCodeBlocks = (markdown: string) => {
  const codeBlockRegex = /```([\s\S]*?)```/g;
  const codeBlocks = [];
  let match;

  while ((match = codeBlockRegex.exec(markdown)) !== null) {
    codeBlocks.push(match[1]);
  }

  return codeBlocks;
};

export const extractParagraphs = (markdown: string) => {
  // Split by double newlines to get paragraphs
  return markdown.split(/\n\s*\n/).filter(p => p.trim() !== '');
};