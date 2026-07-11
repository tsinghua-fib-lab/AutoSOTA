// MalTree project page.

// Copy the BibTeX entry to the clipboard.
function copyBibTeX() {
  const bibtex = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  if (!bibtex || !button) return;
  const label = button.querySelector('.copy-text');

  const restore = () => {
    button.classList.add('copied');
    if (label) label.textContent = 'Cop';
    setTimeout(() => {
      button.classList.remove('copied');
      if (label) label.textContent = 'Copy';
    }, 2000);
  };

  navigator.clipboard.writeText(bibtex.textContent).then(restore).catch(() => {
    const area = document.createElement('textarea');
    area.value = bibtex.textContent;
    document.body.appendChild(area);
    area.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(area);
    restore();
  });
}

// Smooth scroll back to the top of the page.
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Reveal the scroll-to-top button once the page is scrolled.
window.addEventListener('scroll', function () {
  const button = document.querySelector('.scroll-to-top');
  if (button) button.classList.toggle('visible', window.pageYOffset > 300);
});

// Crop the embedded interactive tree to the tree itself.
// The tree viewer draws on a tall canvas; the tree occupies the region below
// in the app's page coordinates (measured at a 1200px-wide viewport).
function fitTree() {
  const wrap = document.getElementById('tree-frame');
  if (!wrap) return;
  const frame = wrap.querySelector('iframe');
  if (!frame) return;
  const X = 24, Y = 446, W = 1160, H = 1122; // tree region in the app page
  const scale = wrap.clientWidth / W;
  frame.style.transformOrigin = 'top left';
  frame.style.transform =
    'scale(' + scale + ') translate(' + (-X) + 'px, ' + (-Y) + 'px)';
  wrap.style.height = (H * scale) + 'px';
}

window.addEventListener('DOMContentLoaded', fitTree);
window.addEventListener('load', fitTree);
window.addEventListener('resize', fitTree);
