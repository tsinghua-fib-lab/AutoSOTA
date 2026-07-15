// MathJax configuration for pymdownx.arithmatex (generic mode).
// Inline math is delimited by \( ... \) and display math by \[ ... \],
// which arithmatex emits; we also accept $...$ / $$...$$ for convenience.
window.MathJax = {
  // Enable the boldsymbol extension (\boldsymbol{...}); it is bundled with the
  // tex-mml-chtml component but not active by default in this generic setup.
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    // tex-mml-chtml does not ship these; define them so the docs' equations render.
    // \coloneqq → :=  ;  \mathbbm{1} → blackboard-bold indicator (same intent as the paper).
    macros: {
      coloneqq: "\\mathrel{:=}",
      eqqcolon: "\\mathrel{=:}",
      mathbbm: ["{\\mathbb{#1}}", 1],
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset on instant navigation (Material's SPA-style page loads).
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
