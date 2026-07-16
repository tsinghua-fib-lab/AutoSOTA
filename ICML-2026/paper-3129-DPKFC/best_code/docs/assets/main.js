/* DP-KFC project page — vanilla JS, no deps */
(() => {
  'use strict';

  /* ---- theme toggle ---- */
  const root = document.documentElement;
  const themeBtn = document.getElementById('themeToggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('dpkfc-theme', next); } catch (e) {}
    });
  }
  // follow the OS theme if the user hasn't explicitly chosen
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
      let saved; try { saved = localStorage.getItem('dpkfc-theme'); } catch (er) {}
      if (saved !== 'light' && saved !== 'dark') root.setAttribute('data-theme', e.matches ? 'light' : 'dark');
    });
  }

  /* ---- scroll progress bar ---- */
  const progress = document.getElementById('progress');
  const nav = document.getElementById('nav');
  const onScroll = () => {
    const h = document.documentElement;
    const scrolled = h.scrollTop / (h.scrollHeight - h.clientHeight || 1);
    if (progress) progress.style.width = (scrolled * 100).toFixed(2) + '%';
    if (nav) nav.classList.toggle('scrolled', h.scrollTop > 12);
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---- reveal on scroll ---- */
  const revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add('visible'));
  }

  /* ---- active nav link ---- */
  const sections = ['problem', 'idea', 'method', 'results', 'why', 'bibtex']
    .map((id) => document.getElementById(id)).filter(Boolean);
  const navLinks = document.querySelectorAll('.nav-links a');
  if ('IntersectionObserver' in window && sections.length) {
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          navLinks.forEach((a) => a.classList.toggle('active',
            a.getAttribute('href') === '#' + e.target.id));
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    sections.forEach((s) => spy.observe(s));
  }

  /* ---- copy buttons ---- */
  document.querySelectorAll('.copybtn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const el = document.getElementById(btn.dataset.target);
      if (!el) return;
      try {
        await navigator.clipboard.writeText(el.innerText.trim());
        const old = btn.textContent;
        btn.textContent = 'Copied ✓'; btn.classList.add('ok');
        setTimeout(() => { btn.textContent = old; btn.classList.remove('ok'); }, 1600);
      } catch { /* clipboard blocked — ignore */ }
    });
  });

  /* ---- graceful handling of missing figure images ---- */
  document.querySelectorAll('.figcard img').forEach((img) => {
    img.addEventListener('error', () => {
      const fig = img.closest('.figcard');
      if (!fig) return;
      img.remove();
      const ph = document.createElement('div');
      ph.style.cssText = 'padding:34px 18px;text-align:center;color:var(--text-faint);font-family:JetBrains Mono,monospace;font-size:.85rem;background:var(--panel-2);';
      ph.textContent = 'figure missing: drop ' + (img.getAttribute('src') || 'image') + ' into docs/static/images/';
      fig.prepend(ph);
    });
  });

  /* ---- title spotlight: a single-colour radial that tracks the cursor ---- */
  (function titleSpotlight(){
    const target = document.querySelector('.title .grad');
    const hero = document.querySelector('.hero');
    if (!target || !hero) return;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0, tx = 50, ty = 45;
    function apply(){ raf = 0; target.style.setProperty('--mx', tx + '%'); target.style.setProperty('--my', ty + '%'); }
    function onMove(e){
      const r = target.getBoundingClientRect();
      tx = ((e.clientX - r.left) / (r.width  || 1)) * 100;
      ty = ((e.clientY - r.top)  / (r.height || 1)) * 100;
      if (tx < -30) tx = -30; else if (tx > 130) tx = 130;
      if (ty < -80) ty = -80; else if (ty > 180) ty = 180;
      if (!raf) raf = requestAnimationFrame(apply);
    }
    if (!reduce) {
      hero.addEventListener('pointermove', onMove, { passive: true });
      hero.addEventListener('pointerleave', () => { tx = 50; ty = 45; apply(); });
    }
  })();

  /* ============================================================
     Hero background: an evolving "structured-noise probe".
     A tiny canvas of cells, each coloured by a cheap value-noise
     field whose spatial frequency slowly breathes (white speckle
     <-> low-frequency structure) — i.e. the 1/f^a shaping DP-KFC
     does to build its probes. Scaled up by CSS with crisp cells.
     ============================================================ */
  (function heroNoise() {
    const canvas = document.getElementById('noiseField');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const hero = document.querySelector('.hero');

    const COLS = 116;                 // internal cell columns; rows derived from aspect
    let cols = COLS, rows = 64;
    let mx = 0.5, my = 0.32;          // pointer, normalised — used for a gentle parallax

    function resize() {
      const r = canvas.getBoundingClientRect();
      const aspect = (r.height || 360) / (r.width || 1000);
      cols = COLS;
      rows = Math.max(10, Math.round(cols * aspect));
      canvas.width = cols;
      canvas.height = rows;
      ctx.imageSmoothingEnabled = false;
    }
    resize();
    let rtimer;
    window.addEventListener('resize', () => { clearTimeout(rtimer); rtimer = setTimeout(resize, 150); }, { passive: true });

    if (hero) hero.addEventListener('pointermove', (e) => {
      const r = hero.getBoundingClientRect();
      mx = (e.clientX - r.left) / (r.width || 1);
      my = (e.clientY - r.top) / (r.height || 1);
    }, { passive: true });

    // monochrome ramp: dark -> active accent. Most cells stay near-black; only
    // the high-value peaks pick up tint -> reads as a heat-map, not a rainbow.
    let TINT = [255, 95, 162];
    function readTint() {
      const v = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
      const m = /^#([0-9a-f]{6})$/i.exec(v);
      if (m) TINT = [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
    }
    readTint();
    if (typeof MutationObserver === 'function') {
      new MutationObserver(readTint).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    }
    function ramp(v) {
      const t = v * v;                            // gamma -> bias toward dark; peaks pop
      return [t * TINT[0], t * TINT[1], t * TINT[2]];
    }

    // value-noise: a handful of drifting sinusoids; `f` (spatial frequency) breathes over ~20s
    function sample(x, y, t, f) {
      const ox = (mx - 0.5) * 0.5, oy = (my - 0.5) * 0.5;
      const u = x + ox, w = y + oy;
      let v = Math.sin(u * f * 6.7 + t * 0.55)
            + Math.sin(w * f * 6.1 - t * 0.40)
            + Math.sin((u + w) * f * 3.9 + t * 0.27)
            + 0.65 * Math.sin((u - w) * f * 10.0 - t * 0.85)
            + 0.45 * Math.sin(u * f * 17.0 + w * f * 14.0 + t * 1.1);  // a touch of fine speckle
      return v / 3.55 * 0.5 + 0.5;       // -> roughly [0,1]
    }

    function draw(t) {
      const f = 0.62 + 0.40 * Math.sin(t * 0.062);   // frequency breath
      for (let j = 0; j < rows; j++) {
        const yy = j / rows;
        for (let i = 0; i < cols; i++) {
          let v = sample(i / cols, yy, t, f);
          if (v < 0) v = 0; else if (v > 1) v = 1;
          const c = ramp(v);
          // alpha tracks intensity: dark cells nearly disappear, bright cells lightly accent
          const a = 0.10 + 0.65 * v;
          ctx.fillStyle = 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + a.toFixed(3) + ')';
          ctx.fillRect(i, j, 1, 1);
        }
      }
    }

    if (reduce) { draw(0); return; }

    let last = -1e9;
    (function loop(ts) {
      requestAnimationFrame(loop);
      if (document.hidden) return;
      if (ts - last < 70) return;          // ~14 fps is plenty for a slow noise field
      last = ts;
      draw(ts * 0.001);
    })(0);
  })();
})();
