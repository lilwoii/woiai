
// electron/main.js
const { app, BrowserWindow, ipcMain, shell } = require("electron");
const fs = require("fs");
const path = require("path");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");

let mainWindow;
let splashWindow;
let backendProcess;

const isDev = !app.isPackaged;

// NOTE: backend runs locally (python). Electron calls backend endpoints for license/OAuth/version.
const API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:8000";

const STORE_FILE = () => path.join(app.getPath("userData"), "woi_store.json");
function readStore() {
  try {
    const p = STORE_FILE();
    if (!fs.existsSync(p)) return {};
    return JSON.parse(fs.readFileSync(p, "utf-8")) || {};
  } catch (e) {
    return {};
  }
}
function writeStore(next) {
  try {
    const p = STORE_FILE();
    fs.writeFileSync(p, JSON.stringify(next || {}, null, 2), "utf-8");
    return true;
  } catch (e) {
    return false;
  }
}



function startBackend() {
  if (backendProcess) return;

  const cwd = isDev ? path.join(__dirname, "..") : process.resourcesPath;
  const py = process.env.WOI_PYTHON || (isDev ? "python" : "python");
  const args = ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", process.env.WOI_BACKEND_PORT || "8000"];
  if (isDev) args.push("--reload");

  backendProcess = spawn(py, args, {
    cwd,
    env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: cwd },
    stdio: "inherit",
  });
  backendProcess.on("close", () => {
    backendProcess = null;
  });
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: "#020617",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  const devURL = process.env.ELECTRON_START_URL || "http://localhost:3000";
  const prodURL = `file://${path.join(__dirname, "..", "frontend", "build", "index.html")}`;
  mainWindow.loadURL(isDev ? devURL : prodURL);

  mainWindow.on("closed", () => (mainWindow = null));
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 860,
    height: 560,
    resizable: false,
    backgroundColor: "#020617",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  splashWindow.loadFile(path.join(__dirname, "splash.html"));
  splashWindow.on("closed", () => (splashWindow = null));
}

function semverCmp(a, b) {
  const pa = String(a || "0.0.0").split(".").map((x) => parseInt(x || "0", 10));
  const pb = String(b || "0.0.0").split(".").map((x) => parseInt(x || "0", 10));
  for (let i = 0; i < 3; i++) {
    const da = pa[i] || 0;
    const db = pb[i] || 0;
    if (da > db) return 1;
    if (da < db) return -1;
  }
  return 0;
}

async function checkUpdate() {
  try {
    const res = await fetch(`${API_BASE_URL}/app/version`, { method: "GET" });
    const data = await res.json();
    const current = app.getVersion();
    const min = data?.min_required || "0.0.0";
    const required = data?.force_update && semverCmp(current, min) < 0;
    return { ok: true, update_required: !!required, download_url: data?.download_url || "" };
  } catch (e) {
    return { ok: false, reason: "offline" };
  }
}

// --- Discord OAuth: local callback listener (scaffold) ---
let oauthServer = null;
function startOAuthListener() {
  if (oauthServer) return;

  oauthServer = http.createServer(async (req, res) => {
    try {
      if (!req.url) return;
      const u = new URL(req.url, "http://127.0.0.1:5179");
      if (u.pathname !== "/discord/callback") {
        res.writeHead(404); res.end("not found"); return;
      }
      const code = u.searchParams.get("code");
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<h3>You can close this window and return to Woi's Assistant.</h3>");

      if (code) {
        await fetch(`${API_BASE_URL}/discord/oauth/exchange`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code }),
        });
      }
    } catch (e) {
      try { res.writeHead(500); res.end("error"); } catch {}
    }
  });

  oauthServer.listen(5179, "127.0.0.1");
}

ipcMain.handle("store:get", async () => {
  return readStore();
});
ipcMain.handle("store:set", async (_evt, patch) => {
  const cur = readStore();
  const next = { ...(cur || {}), ...(patch || {}) };
  const ok = writeStore(next);

  // If webhook provided, persist into backend settings too (best-effort)
  if (patch && patch.discord_webhook_url) {
    try {
      await fetch(`${API_BASE_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ discord_webhook_url: patch.discord_webhook_url }),
      });
    } catch (e) {}
  }

  return { ok, store: next };
});

ipcMain.handle("app:checkUpdate", async () => checkUpdate());

ipcMain.handle("discord:testWebhook", async (_evt, webhookUrl) => {
  try {
    const r = await fetch(`${API_BASE_URL}/discord/test_custom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ webhook_url: webhookUrl }),
    });
    const data = await r.json();
    return { ok: !!data.ok, data };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
});

ipcMain.handle("app:enter", async () => {
  // Enforce license: must be verified and stored
  const store = readStore();
  if (!store.license_key) return { ok: false, error: "Missing license key" };

  try {
    const r = await fetch(`${API_BASE_URL}/license/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: store.license_key }),
    });
    const data = await r.json();
    if (!data.ok) return { ok: false, error: "License invalid" };
  } catch (e) {
    return { ok: false, error: "Backend not ready" };
  }

  createMainWindow();
  if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
  return { ok: true };
});

ipcMain.handle("license:verify", async (evt, { licenseKey }) => {
  try {
    const res = await fetch(`${API_BASE_URL}/license/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ license_key: licenseKey }),
    });
    return await res.json();
  } catch (e) {
    return { ok: false, valid: false, reason: "backend_offline" };
  }
});

ipcMain.handle("discord:startOAuth", async () => {
  try {
    startOAuthListener();
    const res = await fetch(`${API_BASE_URL}/discord/oauth/url`);
    const data = await res.json();
    if (!data?.ok || !data?.url) return { ok: false, reason: data?.reason || "no_url" };
    await shell.openExternal(data.url);
    return { ok: true };
  } catch (e) {
    return { ok: false, reason: "exception" };
  }
});

ipcMain.handle("discord:getProfile", async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/discord/profile`);
    return await res.json();
  } catch (e) {
    return { ok: false, profile: null };
  }
});

ipcMain.handle("app:openMain", async () => {
  if (splashWindow) splashWindow.close();
  if (!mainWindow) createMainWindow();
  return { ok: true };
});

app.whenReady().then(async () => {
  // Start backend first
  startBackend();

  const store = readStore();
  let licenseOk = false;

  // Best-effort license check against backend
  if (store.license_key) {
    try {
      const r = await fetch(`${API_BASE_URL}/license/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: store.license_key }),
      });
      const data = await r.json();
      licenseOk = !!data.ok;
      if (!licenseOk) {
        writeStore({ ...(store || {}), license_ok: false });
      } else {
        writeStore({ ...(store || {}), license_ok: true });
      }
    } catch (e) {
      // If backend isn't up yet, we will still show splash
      licenseOk = false;
    }
  }

  if (licenseOk) {
    createMainWindow();
  } else {
    createSplashWindow();
  }

  app.on("activate", function () {
    if (BrowserWindow.getAllWindows().length === 0) {
      if (licenseOk) createMainWindow();
      else createSplashWindow();
    }
  });
});
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  try { if (oauthServer) oauthServer.close(); } catch {}
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
  }
});