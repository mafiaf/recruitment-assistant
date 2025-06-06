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

  const links = nav.querySelectorAll('ul a');
  const brand = nav.querySelector('a.flex');
  const menuBtn = document.getElementById('menuBtn');
  window.addEventListener('scroll', () => {
    const scrolled = window.scrollY > 50;
    nav.classList.toggle('bg-white', scrolled);
    nav.classList.toggle('bg-opacity-90', scrolled);
    nav.classList.toggle('shadow-lg', scrolled);
    nav.classList.toggle('py-2', scrolled);
    nav.classList.toggle('py-6', !scrolled);
    links.forEach(l => {
      l.classList.toggle('text-gray-800', scrolled);
      l.classList.toggle('text-white', !scrolled);
    });
    if (brand) {
      brand.classList.toggle('text-gray-800', scrolled);
      brand.classList.toggle('text-white', !scrolled);
    }
    if (menuBtn) {
      menuBtn.classList.toggle('text-gray-800', scrolled);
      menuBtn.classList.toggle('text-white', !scrolled);
      menuBtn.classList.toggle('focus:ring-gray-800', scrolled);
      menuBtn.classList.toggle('focus:ring-white', !scrolled);
    }
  });
}

