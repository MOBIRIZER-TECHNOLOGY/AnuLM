// AnuLM-Omni-400M landing page: theme toggle, mobile menu, copy buttons.
// No trackers, no network requests. The page works without this file.
(function () {
  "use strict";
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function current() {
    var t = root.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return media && media.matches ? "dark" : "light";
  }

  var toggle = document.querySelector(".theme-toggle");
  function label() {
    if (!toggle) return;
    var next = current() === "dark" ? "light" : "dark";
    toggle.setAttribute("aria-label", "Switch to " + next + " theme");
    toggle.setAttribute("title", "Switch to " + next + " theme");
  }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("anulm-theme", next); } catch (e) { /* file:// or private mode */ }
      label();
    });
    if (media && media.addEventListener) media.addEventListener("change", label);
    label();
  }

  // Mobile menu
  var menuBtn = document.querySelector(".menu-btn");
  var links = document.getElementById("nav-links");
  function setMenu(open) {
    if (!menuBtn || !links) return;
    links.classList.toggle("open", open);
    menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  }
  if (menuBtn && links) {
    menuBtn.addEventListener("click", function () {
      setMenu(menuBtn.getAttribute("aria-expanded") !== "true");
    });
    links.addEventListener("click", function (e) {
      if (e.target.closest("a")) setMenu(false);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && links.classList.contains("open")) { setMenu(false); menuBtn.focus(); }
    });
  }

  // Copy buttons: copy the commands only, without prompts or comments.
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }
  Array.prototype.forEach.call(document.querySelectorAll(".copy"), function (btn) {
    btn.hidden = false;
    var status = btn.querySelector(".copy-label");
    btn.addEventListener("click", function () {
      var pre = document.getElementById(btn.getAttribute("data-target"));
      if (!pre) return;
      var text = pre.innerText.split("\n")
        .map(function (l) { return l.replace(/^\$ /, ""); })
        .filter(function (l) { return l.trim() && !/^\s*#/.test(l); })
        .join("\n") + "\n";
      function done(ok) {
        if (status) status.textContent = ok ? "Copied" : "Press Ctrl+C";
        setTimeout(function () { if (status) status.textContent = "Copy"; }, 1800);
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallbackCopy(text)); });
      } else {
        done(fallbackCopy(text));
      }
    });
  });
})();
