const header = document.querySelector(".site-header");
const progress = document.querySelector(".scrollbar");
const revealItems = document.querySelectorAll(".reveal");
const canvas = document.getElementById("spectral-canvas");
const ctx = canvas.getContext("2d");

let width = 0;
let height = 0;
let pointerX = 0.5;
let pointerY = 0.5;

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = rect.width;
  height = rect.height;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawSpectralField(time) {
  ctx.clearRect(0, 0, width, height);
  ctx.globalCompositeOperation = "screen";

  const bands = [
    { color: "rgba(94, 234, 212, 0.18)", amp: 34, freq: 0.008, speed: 0.0012 },
    { color: "rgba(96, 165, 250, 0.16)", amp: 24, freq: 0.015, speed: 0.0019 },
    { color: "rgba(167, 139, 250, 0.13)", amp: 16, freq: 0.026, speed: 0.0028 }
  ];

  bands.forEach((band, index) => {
    ctx.beginPath();
    ctx.lineWidth = 1.2 + index * 0.4;
    ctx.strokeStyle = band.color;

    for (let x = -20; x <= width + 20; x += 14) {
      const drift = (pointerX - 0.5) * 70 + index * 46;
      const y =
        height * (0.56 + index * 0.075) +
        Math.sin(x * band.freq + time * band.speed + drift) * band.amp +
        Math.cos(x * band.freq * 0.72 + time * band.speed * 0.78 + pointerY * 4) * band.amp * 0.35;
      if (x === -20) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  });

  ctx.globalCompositeOperation = "source-over";
  requestAnimationFrame(drawSpectralField);
}

function updateScrollState() {
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = maxScroll > 0 ? window.scrollY / maxScroll : 0;
  progress.style.transform = `scaleX(${ratio})`;
  header.classList.toggle("is-scrolled", window.scrollY > 12);
}

const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.16 }
);

revealItems.forEach(item => observer.observe(item));

window.addEventListener("resize", resizeCanvas);
window.addEventListener("scroll", updateScrollState, { passive: true });
window.addEventListener("pointermove", event => {
  pointerX = event.clientX / Math.max(1, window.innerWidth);
  pointerY = event.clientY / Math.max(1, window.innerHeight);
});

window.addEventListener("load", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  resizeCanvas();
  updateScrollState();
  requestAnimationFrame(drawSpectralField);
});
