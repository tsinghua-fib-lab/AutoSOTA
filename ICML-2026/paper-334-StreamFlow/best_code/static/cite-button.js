/**
 * Cite Button Module
 * 为标题添加引用按钮，生成 BibTeX 格式引文
 *
 * 使用方法:
 * 1. 引入 CSS: <link rel="stylesheet" href="./static/cite-button.css">
 * 2. 引入 JS: <script src="./static/cite-button.js"></script>
 * 3. 初始化: CiteButton.init(config);
 */

const CiteButton = (function() {
    'use strict';

    // 默认配置
    const defaultConfig = {
        // BibTeX 配置
        citationType: 'misc',
        author: '{World Snapshot Team}',
        title: 'Related Documents or Contents of Top Level Project (under World Snapshot Template)',
        yearMode: 'current', // 'current' 或具体年份数字

        // 选择器配置
        headerSelector: '.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4, .markdown-body h5, .markdown-body h6',

        // 按钮文本
        buttonText: 'Cite',

        // 样式类名
        citeButtonClass: 'cite-button',
        headerLinkClass: 'header-link',

        // URL 配置
        includeHash: true, // 是否在 URL 中包含标题锚点
    };

    let config = {};

    /**
     * 生成 slug (URL 锚点)
     */
    function generateSlug(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .trim('-');
    }

    /**
     * 获取年份
     */
    function getYear() {
        if (config.yearMode === 'current') {
            return new Date().getFullYear();
        }
        return config.yearMode;
    }

    /**
     * 获取标题的完整层级路径
     */
    function getHeaderPath(headerElement) {
        const path = [];
        const currentLevel = parseInt(headerElement.tagName.substring(1)); // h1 -> 1, h2 -> 2, etc.

        // 获取当前标题文本
        const currentText = getCleanHeaderText(headerElement);
        path.push(currentText);

        // 查找所有在当前标题之前的标题
        const allHeaders = document.querySelectorAll(config.headerSelector);
        let currentIndex = -1;

        // 找到当前标题的索引
        for (let i = 0; i < allHeaders.length; i++) {
            if (allHeaders[i] === headerElement) {
                currentIndex = i;
                break;
            }
        }

        if (currentIndex === -1) return path;

        // 从当前位置向前查找上级标题
        let targetLevel = currentLevel - 1;
        for (let i = currentIndex - 1; i >= 0 && targetLevel >= 1; i--) {
            const header = allHeaders[i];
            const level = parseInt(header.tagName.substring(1));

            if (level === targetLevel) {
                const text = getCleanHeaderText(header);
                path.unshift(text);
                targetLevel--;
            }
        }

        return path;
    }

    /**
     * 获取标题的纯文本（移除按钮等元素）
     */
    function getCleanHeaderText(headerElement) {
        const clone = headerElement.cloneNode(true);
        // 移除所有链接和按钮
        const links = clone.querySelectorAll('a');
        links.forEach(function(link) {
            link.remove();
        });
        return clone.textContent.trim();
    }

    /**
     * 从 URL 中提取前缀（只保留字母）
     * 合并域名和路径的字母部分
     */
    function extractDomainPrefix(url) {
        try {
            let prefix = '';

            // 提取域名部分
            const domainMatch = url.match(/https?:\/\/([^\/]+)/);
            if (domainMatch && domainMatch[1]) {
                let domain = domainMatch[1];
                // 处理端口号：移除 :端口
                domain = domain.replace(/:\d+$/, '');
                // 提取域名中的字母
                const domainLetters = domain.replace(/[^a-zA-Z]/g, '').toLowerCase();
                prefix += domainLetters;
            }

            // 提取路径部分（从域名后到 index.html 或 ? 或 # 之前）
            const pathMatch = url.match(/https?:\/\/[^\/]+\/([^\?#]*)/);
            if (pathMatch && pathMatch[1]) {
                let pathPart = pathMatch[1];
                // 移除 index.html, index.htm 等
                pathPart = pathPart.replace(/\/?index\.html?$/, '');
                // 提取路径中的字母
                const pathLetters = pathPart.replace(/[^a-zA-Z]/g, '').toLowerCase();
                prefix += pathLetters;
            }

            // 如果最终有字母，返回
            if (prefix.length > 0) {
                return prefix;
            }
        } catch (e) {
            console.error('Error extracting domain prefix:', e);
        }
        // 如果提取失败，返回默认值
        return 'worldsnapshotwebsites';
    }

    /**
     * 生成 BibTeX 引文
     */
    function generateBibTeX(title, url, headerElement) {
        const year = getYear();

        // 从 URL 提取域名前缀（只保留字母）
        const domainPrefix = extractDomainPrefix(url);
        const titleSlug = generateSlug(title);
        const citationKey = domainPrefix + '_' + titleSlug;

        // 获取完整的标题路径
        const fullPath = headerElement ? getHeaderPath(headerElement).join(' > ') : title;

        // 获取干净的标题文本用于 title 字段
        const actualTitle = headerElement ? getCleanHeaderText(headerElement) : title;

        return `@${config.citationType}{${citationKey},
  author = {${config.author}},
  title = {${actualTitle} (under World Snapshot Template)},
  year = {${year}},
  url = {${url}},
  note = {Section: ${fullPath}}
}`;
    }

    /**
     * HTML 转义
     */
    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * 显示 BibTeX 引文窗口
     */
    function showBibTeX(title, url, headerElement) {
        const bibtex = generateBibTeX(title, url, headerElement);
        const escapedBibtex = escapeHtml(bibtex);

        const htmlContent = [
            '<!DOCTYPE html>',
            '<html lang="en" class="translate">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<meta http-equiv="Content-Language" content="en">',
            '<meta name="description" content="BibTeX citation for ' + escapeHtml(title) + '">',
            '<title>BibTeX Citation - ' + escapeHtml(title) + '</title>',
            '<style>',
            '* { margin: 0; padding: 0; box-sizing: border-box; }',
            'html, body {',
            '  height: 100%;',
            '}',
            'body {',
            '  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;',
            '  line-height: 1.6;',
            '  color: #202122;',
            '  background: #f8f9fa;',
            '  padding: 20px;',
            '  display: flex;',
            '  flex-direction: column;',
            '  min-height: 100vh;',
            '}',
            '.container {',
            '  max-width: 1000px;',
            '  margin: 0 auto;',
            '  background: #fff;',
            '  padding: 40px;',
            '  border: 1px solid #a2a9b1;',
            '  flex: 1;',
            '}',
            'h1 {',
            '  font-size: 1.8em;',
            '  margin-bottom: 0.8em;',
            '  padding-bottom: 0.3em;',
            '  border-bottom: 1px solid #a2a9b1;',
            '  color: #000;',
            '  font-family: "Linux Libertine", "Georgia", "Times", serif;',
            '}',
            'h2 {',
            '  font-size: 1.5em;',
            '  margin: 1em 0;',
            '  padding-bottom: 0.2em;',
            '  border-bottom: 1px solid #c8ccd1;',
            '  color: #162441;',
            '  font-family: "Linux Libertine", "Georgia", "Times", serif;',
            '}',
            'p {',
            '  margin: 1em 0;',
            '  line-height: 1.6;',
            '}',
            'ol {',
            '  padding-left: 0;',
            '  margin: 1em 0 1em 16px;',
            '  list-style-position: outside;',
            '}',
            'li {',
            '  margin: 0.5em 0;',
            '}',
            'pre {',
            '  margin: 1em 0;',
            '  padding: 16px;',
            '  overflow: auto;',
            '  background-color: #f6f8fa;',
            '  border-radius: 3px;',
            '  font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace;',
            '  font-size: 14px;',
            '  line-height: 1.6;',
            '  border: 1px solid #d0d7de;',
            '  white-space: pre-wrap;',
            '  word-wrap: break-word;',
            '}',
            '.copy-btn {',
            '  background-color: #3366cc;',
            '  color: white;',
            '  border: none;',
            '  padding: 10px 20px;',
            '  border-radius: 3px;',
            '  cursor: pointer;',
            '  font-size: 14px;',
            '  margin-top: 15px;',
            '  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;',
            '}',
            '.copy-btn:hover { background-color: #2952a3; }',
            '.success-message {',
            '  color: #28a745;',
            '  margin-top: 10px;',
            '  display: none;',
            '  font-size: 14px;',
            '}',
            'footer {',
            '  font-size: small;',
            '  border-top: 1px solid #c0c0c0;',
            '  padding-top: 0.1em;',
            '  margin-top: 0.1em;',
            '  color: #c0c0c0;',
            '  max-width: 1000px;',
            '  width: 100%;',
            '  margin-left: auto;',
            '  margin-right: auto;',
            '}',
            'footer a {',
            '  color: #80a0b0;',
            '  text-decoration: none;',
            '}',
            'footer a:hover {',
            '  text-decoration: underline;',
            '}',
            '.footer-text {',
            '  padding-left: 40px;',
            '  padding-bottom: 0;',
            '  text-align: center;',
            '}',
            '</style>',
            '</head>',
            '<body>',
            '<div class="container">',
            '<h1>BibTeX Citation</h1>',
            '<pre id="bibtex" translate="no">' + escapedBibtex + '</pre>',
            '<button class="copy-btn" id="copyBtn">Copy to Clipboard</button>',
            '<div class="success-message" id="successMsg">Copied to clipboard!</div>',
            '<h2>Note</h2>',
            '<p>We suggest not modifying the automatically generated BibTeX:</p>',
            '<ol>',
            '<li>Generally, the key should not be modified. If the generated Key conflicts, for example, if two websites both use our template and both have an Overview, then you can manually modify it. This will not affect the correct citation of your content. This is quite rare.</li>',
            '<li>For Author, since the contribution of the Doc is uncertain and the number of authors will increase, it is best to use a team name. This is also beneficial for you to increase citations and attract more contributors to contribute to your project. And a fixed name is convenient for searching.</li>',
            '<li>For the title, the default is to use the current paragraph title that is being referenced. It\'s better not to remove "(under World Snapshot Template)" after the title, because if you don\'t include this, it will be very difficult for you to quickly search for who has cited you on Google. This is equivalent to a filter identifier.</li>',
            '<li>The year is automatically obtained from the current year of the Doc. We assume that you will always maintain and be responsible for the content you publish online.</li>',
            '<li>The URL and note are automatically obtained from the current title of the cite.</li>',
            '</ol>',
            '<p>In summary, anyone involved in compiling the documents should search independently to add these citations to the Google Scholar homepage (As long as the contributor has participated in the revision of the document, he have the right to include this citation in his Google Scholar profile).</p>',
            '<p><strong>Important for Template Users:</strong> If you cloned this World Snapshot Doc Template for your own project, please remember to modify the <code>author</code> field in <code>doc/index.html</code> (line 528) to reflect your project\'s team name. For example, change <code>author: \'{World Snapshot Team}\'</code> to <code>author: \'{Your Project Team}\'</code>. This ensures proper attribution for your project.</p>', 
            '</div>',
            '<footer>',
            '<div class="footer-text">',
            '© CC BY-SA 4.0: Feel free to use this template, but please keep the Powered by <a href="https://github.com/World-Snapshot/doc" target="_blank">World Snapshot Doc</a>.',
            '</div>',
            '</footer>',
            '<script>',
            'document.getElementById("copyBtn").onclick = function() {',
            '  var text = document.getElementById("bibtex").textContent;',
            '  navigator.clipboard.writeText(text).then(function() {',
            '    var msg = document.getElementById("successMsg");',
            '    msg.style.display = "block";',
            '    setTimeout(function() { msg.style.display = "none"; }, 2000);',
            '  }).catch(function(err) { alert("Failed to copy: " + err); });',
            '};',
            '</script>',
            '</body>',
            '</html>'
        ].join('\n');

        // 创建真实的 HTML Blob，浏览器会将其视为真实网页
        const blob = new Blob([htmlContent], { type: 'text/html; charset=utf-8' });
        const blobUrl = URL.createObjectURL(blob);

        // 打开 Blob URL，这是一个真实的 HTML 文件
        const newWindow = window.open(blobUrl, '_blank');

        // 清理 URL（延迟清理，确保页面加载完成）
        if (newWindow) {
            setTimeout(function() {
                URL.revokeObjectURL(blobUrl);
            }, 1000);
        }
    }

    /**
     * 为标题添加 Cite 按钮
     */
    function addCiteButtons() {
        const headers = document.querySelectorAll(config.headerSelector);

        headers.forEach(function(header) {
            const text = header.textContent;
            const slug = generateSlug(text);

            // 设置标题 ID
            if (!header.id) {
                header.id = slug;
            }

            // 检查是否已经添加过按钮
            if (header.querySelector('.' + config.citeButtonClass)) {
                return;
            }

            // 创建 Cite 按钮
            const citeButton = document.createElement('a');
            citeButton.className = config.citeButtonClass;
            citeButton.textContent = config.buttonText;
            citeButton.href = '#';
            citeButton.setAttribute('aria-label', 'Cite this section');

            citeButton.onclick = function(e) {
                e.preventDefault();
                const baseUrl = window.location.origin + window.location.pathname + window.location.search;
                const fullUrl = config.includeHash ? baseUrl + '#' + slug : baseUrl;
                showBibTeX(text, fullUrl, header);
            };

            // 将按钮插入到标题中
            header.appendChild(citeButton);
        });
    }

    /**
     * 初始化模块
     */
    function init(userConfig) {
        // 合并配置
        config = Object.assign({}, defaultConfig, userConfig || {});

        // 添加 Cite 按钮
        addCiteButtons();

        return {
            config: config,
            refresh: addCiteButtons // 暴露刷新方法，用于动态内容
        };
    }

    // 公开 API
    return {
        init: init,
        generateBibTeX: function(title, url, headerElement) {
            return generateBibTeX(title, url, headerElement);
        },
        version: '1.3.1'
    };
})();

// 如果在 Node.js 环境中，导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CiteButton;
}
