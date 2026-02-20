// ═══════════════════════════════════════════════════════════════
// main.js — Electron wrapper for Prompt Builder IDE
// Spawns the Python FastAPI backend, then loads http://127.0.0.1:8000
// ═══════════════════════════════════════════════════════════════

const { app, BrowserWindow, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const API_PORT = 8000;
const API_URL = `http://127.0.0.1:${API_PORT}`;

let pyProcess = null;
let mainWindow = null;

// ── Spawn Python backend ──
function startBackend() {
    const projectRoot = path.resolve(__dirname, '..');
    pyProcess = spawn('python', [
        '-m', 'uvicorn',
        'src.api.main:app',
        '--host', '127.0.0.1',
        '--port', String(API_PORT),
    ], {
        cwd: projectRoot,
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    pyProcess.stdout.on('data', (data) => {
        console.log(`[uvicorn] ${data.toString().trim()}`);
    });

    pyProcess.stderr.on('data', (data) => {
        console.error(`[uvicorn] ${data.toString().trim()}`);
    });

    pyProcess.on('close', (code) => {
        console.log(`[uvicorn] exited with code ${code}`);
        pyProcess = null;
    });

    pyProcess.on('error', (err) => {
        dialog.showErrorBox(
            'Backend Error',
            `Failed to start Python backend:\n${err.message}\n\nMake sure Python and all requirements are installed.`
        );
    });
}

// ── Wait for backend to be ready ──
function waitForBackend(retries = 30, interval = 1000) {
    return new Promise((resolve, reject) => {
        let attempts = 0;
        const check = () => {
            attempts++;
            const req = http.get(`${API_URL}/api/health`, (res) => {
                if (res.statusCode === 200) {
                    resolve();
                } else if (attempts < retries) {
                    setTimeout(check, interval);
                } else {
                    reject(new Error('Backend did not respond after ' + retries + ' attempts'));
                }
            });
            req.on('error', () => {
                if (attempts < retries) {
                    setTimeout(check, interval);
                } else {
                    reject(new Error('Backend did not start'));
                }
            });
            req.end();
        };
        check();
    });
}

// ── Create window ──
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 600,
        title: 'Prompt Builder — KAIJU STATION',
        backgroundColor: '#070b10',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    mainWindow.loadURL(API_URL);

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// ── App lifecycle ──
app.on('ready', async () => {
    startBackend();
    try {
        await waitForBackend();
    } catch (e) {
        dialog.showErrorBox('Startup Error', e.message);
    }
    createWindow();
});

app.on('window-all-closed', () => {
    if (pyProcess) {
        pyProcess.kill();
        pyProcess = null;
    }
    app.quit();
});

app.on('before-quit', () => {
    if (pyProcess) {
        pyProcess.kill();
        pyProcess = null;
    }
});
