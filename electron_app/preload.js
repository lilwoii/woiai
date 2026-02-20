
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  checkUpdate: () => ipcRenderer.invoke("app:checkUpdate"),
  storeGet: () => ipcRenderer.invoke("store:get"),
  storeSet: (patch) => ipcRenderer.invoke("store:set", patch),
  verifyLicense: (licenseKey) => ipcRenderer.invoke("license:verify", { licenseKey }),
  startDiscordOAuth: () => ipcRenderer.invoke("discord:startOAuth"),
  getDiscordProfile: () => ipcRenderer.invoke("discord:getProfile"),
  openMain: () => ipcRenderer.invoke("app:openMain"),
});
