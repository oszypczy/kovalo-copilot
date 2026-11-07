// Kovalo Decision Copilot: panel boczny (ścieżka dowodowa, koszt tury, fragmenty) otwiera się
// tylko po kliknięciu linku w wiadomości. Chainlit 2.12 sam otwiera panel, gdy w rozmowie pojawi
// się nowy element boczny; ten skrypt (custom_js w .chainlit/config.toml) zamyka go wtedy od razu.
(function () {
  const CLICK_WINDOW_MS = 1500;
  let lastLinkClick = 0;

  document.addEventListener(
    "click",
    (event) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target && target.closest("a.element-link")) {
        lastLinkClick = Date.now();
      }
    },
    true
  );

  let lastTitle = null;
  const observer = new MutationObserver(() => {
    const title = document.getElementById("side-view-title");
    const text = title ? title.textContent : null;
    if (text === lastTitle) return;
    lastTitle = text;
    if (!title || Date.now() - lastLinkClick < CLICK_WINDOW_MS) return;
    const close = title.querySelector("button");
    if (close) close.click();
  });

  const start = () => observer.observe(document.body, { childList: true, subtree: true });
  if (document.body) start();
  else document.addEventListener("DOMContentLoaded", start);
})();
