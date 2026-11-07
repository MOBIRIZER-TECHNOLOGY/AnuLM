// AnuLM-Omni-400M landing page: theme toggle, mobile menu, copy buttons, enquiry form.
// No trackers. The only network request is the enquiry itself, sent when a visitor
// presses "Send" to the form service set in the form's data-endpoint (docs/LAUNCH.md).
// The page works without this file; the form then does nothing.
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

  // Enquiry form. Configured by attributes on <form id="enquiry-form">:
  //   data-endpoint    a form service URL that accepts a JSON POST (Web3Forms, Formspree, ...)
  //   data-access-key  the service's public key, if it wants one in the body (Web3Forms)
  //   data-email       fallback: open the visitor's mail app addressed here
  var form = document.getElementById("enquiry-form");
  // The form ships hidden, with a public contact card in its place, and appears only
  // once it is connected: until then the page collects no personal data at all.
  if (form && (form.getAttribute("data-endpoint") || form.getAttribute("data-email"))) {
    form.hidden = false;
    var fallback = document.getElementById("contact-fallback");
    if (fallback) fallback.hidden = true;
    Array.prototype.forEach.call(document.querySelectorAll("[data-requires-form]"), function (el) { el.hidden = false; });
  } else {
    form = null;
  }
  if (form) {
    var status = form.querySelector(".form-status");
    var say = function (msg, kind) {
      status.textContent = msg;
      status.className = "form-status" + (kind ? " " + kind : "");
    };
    Array.prototype.forEach.call(document.querySelectorAll("a[data-topic]"), function (a) {
      a.addEventListener("click", function () {
        var sel = form.elements.topic;
        if (sel) sel.value = a.getAttribute("data-topic");
      });
    });
    var invalid = function () {
      var bad = [];
      Array.prototype.forEach.call(form.querySelectorAll("[required]"), function (el) {
        var ok = el.type === "checkbox" ? el.checked : el.value.trim() !== "";
        if (ok && el.type === "email") ok = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(el.value.trim());
        el.setAttribute("aria-invalid", ok ? "false" : "true");
        if (!ok) bad.push(el);
      });
      return bad;
    };
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var bad = invalid();
      if (bad.length) {
        say(bad.some(function (el) { return el.name === "adult"; }) && bad.length === 1
          ? "Please confirm you are 18 or older. Under 18? Ask a teacher or parent to write to us."
          : "Please fill in the highlighted fields and tick both boxes.", "err");
        bad[0].focus();
        return;
      }
      var f = form.elements;
      if (f.website && f.website.value) { say("Thank you. We will reply by email.", "ok"); form.reset(); return; }
      var data = {
        name: f.name.value.trim(), email: f.email.value.trim(), role: f.role.value,
        topic: f.topic.value, organisation: f.organisation.value.trim(), message: f.message.value.trim(),
        subject: "AnuLM enquiry: " + f.topic.value + " (" + f.role.value + ")",
        consent: "Agreed to the privacy notice; confirmed 18 or older"
      };
      var endpoint = form.getAttribute("data-endpoint");
      var key = form.getAttribute("data-access-key");
      var email = form.getAttribute("data-email");
      var mailto = function () {
        var body = "Name: " + data.name + "\nEmail: " + data.email + "\nI am: " + data.role +
          "\nInterested in: " + data.topic + "\nOrganisation: " + data.organisation + "\n\n" + data.message;
        window.location.href = "mailto:" + email + "?subject=" + encodeURIComponent(data.subject) +
          "&body=" + encodeURIComponent(body);
        say("Your mail app should open with the enquiry filled in. If it does not, write to " + email + ".", "ok");
      };
      if (endpoint) {
        if (key) data.access_key = key;
        var btn = form.querySelector("button[type=submit]");
        btn.disabled = true;
        say("Sending…");
        fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json", "Accept": "application/json" },
                          body: JSON.stringify(data) })
          .then(function (r) { if (!r.ok) throw new Error(r.status); })
          .then(function () { say("Thank you. Your enquiry was sent; we will reply by email.", "ok"); form.reset(); })
          .catch(function () {
            if (email) { mailto(); } else { say("Sending failed. Please try again, or ask on GitHub.", "err"); }
          })
          .then(function () { btn.disabled = false; });
      } else if (email) {
        mailto();
      } else {
        say("The form is not connected yet. Please open an issue on GitHub; we read every one.", "err");
      }
    });
  }
})();
