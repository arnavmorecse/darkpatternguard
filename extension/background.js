// DarkPatternGuard background worker: talks to the local API and keeps the toolbar badge.
const API = "http://127.0.0.1:8765";
const BADGE = { high: "#D6293E", medium: "#E8710A", low: "#C99A06", privacy: "#7A4FD0", none: "#1E8E5A" };

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "classify") {
    fetch(API + "/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: msg.texts })
    })
      .then(r => r.json())
      .then(d => sendResponse({ ok: true, ...d }))
      .catch(e => sendResponse({ ok: false, error: String(e) }));
    return true; // async response
  }
  if (msg.type === "health") {
    fetch(API + "/health").then(r => r.json())
      .then(d => sendResponse({ ok: true, ...d }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
  if (msg.type === "badge" && sender.tab) {
    const tabId = sender.tab.id;
    const text = msg.offline ? "!" : msg.count ? String(msg.count) : "";
    chrome.action.setBadgeText({ tabId, text });
    chrome.action.setBadgeBackgroundColor({ tabId, color: msg.offline ? "#8A9099" : BADGE[msg.worst || "none"] });
    if (chrome.action.setBadgeTextColor) chrome.action.setBadgeTextColor({ tabId, color: "#FFFFFF" });
  }
});
