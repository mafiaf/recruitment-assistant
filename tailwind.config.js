/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    // main templates
    "./templates/**/*.html",
    "./templates_en/**/*.html",

    // any nested language or fragment folders
    "./templates_*/*.html",

    // if you embed classes in Python strings
    "./**/*.py",

    // JS that contains class-names
    "./static/**/*.js"
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Manrope", "sans-serif"],
      },
    },
  },
  plugins: [],
};
