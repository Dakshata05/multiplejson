
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow;
let splash;

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  });
  mainWindow.loadFile('index.html');

  // Create splash screen
  splash = new BrowserWindow({ 
    width: 800, 
    height: 600, 
    transparent: true, 
    frame: false, 
    alwaysOnTop: true,
    show: false
  });
  splash.loadFile('splash.html');
  
  splash.center();

  // Show splash screen
  splash.show();

  setTimeout(function () {
    splash.close();
    mainWindow.center();
    mainWindow.show();
  }, 5000);

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// Listen for the event from renderer process
ipcMain.on('show-splash', () => {
  splash.show();
});


