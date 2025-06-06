import AOS from 'https://unpkg.com/aos@2.3.4/dist/aos.js';
import { gsap } from 'https://cdn.jsdelivr.net/npm/gsap@3.12.5/+esm';

AOS.init({ once: true, duration: 600 });

// Example hero animation timeline
export function heroTimeline(selector) {
  return gsap.timeline().from(selector, { opacity: 0, y: 50, duration: 1 });
}

