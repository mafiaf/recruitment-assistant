import AOS from 'https://unpkg.com/aos@2.3.4/dist/aos.js';
import { gsap } from 'https://cdn.jsdelivr.net/npm/gsap@3.12.5/+esm';

AOS.init({ once: true, duration: 600 });

// Example hero animation timeline
export function heroTimeline(selector) {
  return gsap.timeline().from(selector, { opacity: 0, y: 50, duration: 1 });
}

document.addEventListener('DOMContentLoaded', () => {
  const hero = document.querySelector('section.relative.h-screen');
  if (hero) {
    heroTimeline(hero.querySelectorAll('h1, p, #uploadCTA'));
  }
});

const mobileBtn = document.getElementById('menuBtn');
const mobileMenu = document.getElementById('mobileMenu');
if (mobileBtn && mobileMenu) {
  mobileBtn.addEventListener('click', () => {
    mobileMenu.classList.toggle('-translate-y-full');
  });
  mobileMenu.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => mobileMenu.classList.add('-translate-y-full'));
  });
}

const nav = document.getElementById('mainNav');
if (nav) {
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      nav.classList.add('bg-white', 'bg-opacity-90', 'shadow-lg', 'py-2');
      nav.classList.remove('py-6');
    } else {
      nav.classList.remove('bg-white', 'bg-opacity-90', 'shadow-lg', 'py-2');
      nav.classList.add('py-6');
    }
  });
}

