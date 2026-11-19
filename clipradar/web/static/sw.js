// Minimal service worker: makes ClipRadar installable and caches the app shell.
const SHELL = "clipradar-v1";
self.addEventListener("install", e => e.waitUntil(caches.open(SHELL).then(c => c.addAll(["/", "/icon.svg", "/font.otf"])).then(() => self.skipWaiting())));
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.pathname.startsWith("/api/") || u.pathname.startsWith("/files/")) return;
  e.respondWith(fetch(e.request).then(r => { const copy = r.clone(); caches.open(SHELL).then(c => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request)));
});
