..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Research Paper
==============

This page presents the full research paper which is also available on `arXiv <https://arxiv.org/abs/2605.28612>`.

.. raw:: html

   <link rel="stylesheet" type="text/css" href="assets/paper_style.css">
   <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js" type="text/javascript"></script>

.. raw:: html
   :file: paper_content.html
   :class: paper-content

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', function () {
     var headings = document.querySelectorAll('h1[data-number], h2[data-number], h3[data-number]');
     if (headings.length === 0) return;

     function getHeadingText(h) {
       var clone = h.cloneNode(true);
       var numSpan = clone.querySelector('.header-section-number');
       if (numSpan) numSpan.remove();
       return clone.textContent.trim();
     }

     // Root item: "Paper" → link to #paper (Sphinx page title anchor)
     var rootUl = document.createElement('ul');
     var rootLi = document.createElement('li');
     var rootA = document.createElement('a');
     rootA.className = 'reference internal';
     rootA.href = '#paper';
     rootA.textContent = 'Paper';
     rootLi.appendChild(rootA);
     rootUl.appendChild(rootLi);

     // Build nested TOC
     var stack = [{ level: 0, li: rootLi }];
     headings.forEach(function (h) {
       var level = parseInt(h.tagName.charAt(1));
       var id = h.getAttribute('id');
       var text = getHeadingText(h);

       var li = document.createElement('li');
       var a = document.createElement('a');
       a.className = 'reference internal';
       a.href = '#' + id;
       a.textContent = text;
       li.appendChild(a);

       while (stack.length > 1 && stack[stack.length - 1].level >= level) {
         stack.pop();
       }
       var parentLi = stack[stack.length - 1].li;
       var parentUl = parentLi.querySelector(':scope > ul');
       if (!parentUl) {
         parentUl = document.createElement('ul');
         parentLi.appendChild(parentUl);
       }
       parentUl.appendChild(li);
       stack.push({ level: level, li: li });
     });

     // Inject into Furo's toc-drawer
     var tocDrawer = document.querySelector('.toc-drawer');
     if (!tocDrawer) return;
     tocDrawer.classList.remove('no-toc');

     var tocSticky = document.createElement('div');
     tocSticky.className = 'toc-sticky toc-scroll';

     var tocTitleContainer = document.createElement('div');
     tocTitleContainer.className = 'toc-title-container';
     var tocTitle = document.createElement('span');
     tocTitle.className = 'toc-title';
     tocTitle.textContent = 'On this page';
     tocTitleContainer.appendChild(tocTitle);

     var tocTreeContainer = document.createElement('div');
     tocTreeContainer.className = 'toc-tree-container';
     var tocTree = document.createElement('div');
     tocTree.className = 'toc-tree';
     tocTree.appendChild(rootUl);
     tocTreeContainer.appendChild(tocTree);

     tocSticky.appendChild(tocTitleContainer);
     tocSticky.appendChild(tocTreeContainer);
     tocDrawer.appendChild(tocSticky);

     // Remove no-toc from TOC icon labels
     document.querySelectorAll('.no-toc').forEach(function (el) {
       el.classList.remove('no-toc');
     });

     // Wrap QED markers (◻) to float right
     var walker = document.createTreeWalker(
       document.querySelector('.paper-content'),
       NodeFilter.SHOW_TEXT,
       null,
       false
     );

     var nodesToReplace = [];
     var node;
     while (node = walker.nextNode()) {
       if (node.textContent.includes('◻')) {
         nodesToReplace.push(node);
       }
     }

     nodesToReplace.forEach(function (textNode) {
       var parent = textNode.parentNode;
       var html = parent.innerHTML;
       html = html.replace(/◻/g, '<span class="proof-qed">◻</span>');
       parent.innerHTML = html;
     });
   });
   </script>

