#!/usr/bin/env python3
"""
Xbox One auto-updater (server side).

Pipeline:
  HDMI capture card -> OpenCV frames -> OCR -> state machine -> HTTP key
  commands -> ESP32-S3 (USB HID keyboard) -> Xbox One

The ESP32 firmware is expected to expose:
    POST {ESP_URL}/key   body: {"key": "down"}      (one tap)
Key names used here: up, down, left, right, enter, esc, tab, guide

Usage:
    python xbox_updater.py --list-devices          # show usable video inputs
    python xbox_updater.py --view-only             # live view only, pick input in browser
    python xbox_updater.py --live                  # run updater AND serve live view
    python xbox_updater.py --device 2 --live       # choose the input at startup
    python xbox_updater.py --ocr-test              # print what OCR sees right now
    python xbox_updater.py --snapshot out.png      # save one frame
    python xbox_updater.py --dry-run               # run flow, print keys instead of sending
    python xbox_updater.py --setup --live          # get through first-time Xbox setup
    python xbox_updater.py --setup --then-update   # ...then run the system update check
    python xbox_updater.py --write-default-rules setup_rules.json   # editable setup rules
    python xbox_updater.py --setup --rules setup_rules.json
    python xbox_updater.py --save-template welcome --roi 400,100,480,80   # OpenCV template
    python xbox_updater.py --check-rules           # keep checking the screen against the rules until Ctrl+C
    python xbox_updater.py --check-rules --check-once   # old behaviour: check once and exit
    python xbox_updater.py --calibrate             # learn setup rules while you click through setup
    python xbox_updater.py --live --rules my_rules.json   # live page has a Learning mode panel that
                                                   # adds rules to my_rules.json (no hand editing)

Visit counting: during --setup the script counts how many times each screen (rule) has been seen,
logs "Screen 'x' seen N time(s)" every time, and writes xbox_logs/screen_counts.json. A rule can press
different keys on a given visit ("visit_keys": {"2": [["tab", 1], ["enter", 1]]}); in Learning mode
fill in "Only on visit number" to add that. --visit-similarity tunes what counts as a new visit.

Live view: open http://<server-ip>:8080 in any browser (works headless).
Menu text, key paths and timings are best guesses for the current dashboard.
Run --ocr-test on each screen and adjust the FLOW / *_TEXT constants below.
"""

import argparse
import difflib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide V4L2 probe warnings; must precede cv2 import
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytesseract  # noqa: E402
import requests  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ESP_URL = "http://xbox-keyboard.local"   # or the ESP32's IP, e.g. http://192.168.1.50
CAPTURE_INDEX = 0                        # default video input (see --list-devices)
CAPTURE_SIZE = (1280, 720)
LIVE_PORT = 8080
STREAM_FPS = 15
KEY_DELAY = 0.6                          # seconds to let UI animations settle after a key
SCREEN_TIMEOUT = 20                      # seconds to wait for an expected screen
UPDATE_TIMEOUT = 60 * 60                 # max seconds to wait for an update to finish
MATCH_THRESHOLD = 0.82                   # fuzzy match ratio for OCR text
LOG_DIR = Path("xbox_logs")
TEMPLATE_DIR = Path("templates")         # OpenCV reference images for setup screens
PAUSE_TIMEOUT = 30 * 60                  # how long to wait for you on a "needs you" screen
UNMATCHED_TIMEOUT = 90                   # seconds on an unrecognized screen before asking you
VISIT_SIMILARITY = 0.85                  # same rule + screen text this similar = same visit
                                         # (lower = more tolerant of OCR noise; see --visit-similarity)

# Phrases to look for (lowercase). Tune with --ocr-test.
UPDATE_AVAILABLE_TEXT = ["console update available", "update available", "update now"]
UP_TO_DATE_TEXT = ["no console update available", "your console is up to date",
                   "no updates available", "up to date"]
UPDATING_TEXT = ["updating", "installing", "downloading", "% complete", "preparing"]
RESTART_TEXT = ["restart", "restarting"]
ERROR_TEXT = ["something went wrong", "couldn't", "could not", "error", "0x8"]

log = logging.getLogger("xbox-updater")

# Shared status shown in the live view overlay / page.
STATUS = {"state": "idle", "detail": "", "ocr": ""}


def set_status(state: str, detail: str = "") -> None:
    STATUS["state"] = state
    STATUS["detail"] = detail


def set_ocr(text: str) -> None:
    """Latest text read from the screen; shown under the video on the live page."""
    STATUS["ocr"] = text[:400]


# How many separate times each setup screen (rule) has been seen during this run.
# A "visit" is one appearance: leaving the screen and coming back is a new visit, and so is
# the same rule suddenly showing visibly different content (e.g. a second sign-in page).
VISITS: dict = {}


def save_visit_counts() -> None:
    """Write the visit counts to xbox_logs/screen_counts.json (best effort)."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "screen_counts.json").write_text(json.dumps(VISITS, indent=2), encoding="utf-8")
    except OSError as e:
        log.debug("Could not write screen_counts.json: %s", e)


def log_visit_summary() -> None:
    if not VISITS:
        return
    log.info("Screen visit counts: %s",
             ", ".join(f"{n} x{c}" for n, c in sorted(VISITS.items(), key=lambda kv: -kv[1])))
    save_visit_counts()


# ----------------------------------------------------------------------------
# Key sender
# ----------------------------------------------------------------------------
class KeySender:
    def __init__(self, base_url: str, dry_run: bool = False, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run
        self.session = requests.Session()
        if token:
            self.session.headers["X-Token"] = token   # matches AUTH_TOKEN in the firmware
        self._lock = threading.Lock()                 # updater + web remote share one sender
        self.history: list = []                       # (timestamp, key) of every press

    def keys_between(self, t0: float, t1: float) -> list[str]:
        return [k for t, k in self.history if t0 < t <= t1]

    def press(self, key: str, count: int = 1, delay: float = KEY_DELAY) -> None:
        for _ in range(count):
            with self._lock:
                self.history.append((time.time(), key))
                if self.dry_run:
                    log.info("[dry-run] key: %s", key)
                else:
                    for attempt in range(3):
                        try:
                            r = self.session.post(f"{self.base_url}/key",
                                                  json={"key": key}, timeout=3)
                            r.raise_for_status()
                            break
                        except requests.RequestException as e:
                            log.warning("key %s failed (attempt %d): %s", key, attempt + 1, e)
                            time.sleep(1)
                    else:
                        raise RuntimeError(f"Could not send key '{key}' to {self.base_url}")
                    log.info("key: %s", key)
            time.sleep(delay)


# ----------------------------------------------------------------------------
# Video devices
# ----------------------------------------------------------------------------
def _backend():
    return cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY


def list_devices(max_index: int = 12) -> list[dict]:
    """Probe video inputs that actually deliver frames. Call before opening any."""
    found = []
    for i in range(max_index):
        if sys.platform.startswith("linux") and not Path(f"/dev/video{i}").exists():
            continue
        cap = cv2.VideoCapture(i, _backend())
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            continue
        name = f"Video {i}"
        name_file = Path(f"/sys/class/video4linux/video{i}/name")
        if name_file.exists():
            name = f"{name_file.read_text().strip()} (video{i})"
        found.append({"index": i, "name": name})
    return found


# ----------------------------------------------------------------------------
# Video capture + OCR
# ----------------------------------------------------------------------------
class Screen:
    """Continuously reads the selected input in a background thread so the
    updater and the live stream can both grab the latest frame safely."""

    def __init__(self, index: int, size=(1920, 1080), save_dir: Path | None = None):
        self.size = size
        self.index = index
        self.save_dir = save_dir
        self._shot = 0
        self._latest = None
        self._lock = threading.Lock()
        self._switch_lock = threading.Lock()
        self._cap = None
        self._stop = threading.Event()
        self._thread = None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        self._start(index)

    # -- capture thread ------------------------------------------------------
    def _open(self, index: int):
        cap = cv2.VideoCapture(index, _backend())
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open capture device {index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _start(self, index: int) -> None:
        self._cap = self._open(index)
        self.index = index
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop,
                                        args=(self._cap, self._stop), daemon=True)
        self._thread.start()

    def _halt(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()

    def _loop(self, cap, stop: threading.Event) -> None:
        while not stop.is_set():
            ok, f = cap.read()
            if ok and f is not None:
                with self._lock:
                    self._latest = f
            else:
                time.sleep(0.05)

    def switch(self, index: int) -> None:
        """Change the video input on the fly."""
        with self._switch_lock:
            if index == self.index:
                return
            old = self.index
            self._halt()
            try:
                self._start(index)
                log.info("Switched video input %d -> %d", old, index)
            except Exception:
                self._start(old)   # fall back to the previous input
                raise

    # -- frames --------------------------------------------------------------
    def frame(self, timeout: float = 5.0):
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.02)
        raise RuntimeError("No frame from capture device")

    def is_black(self, frame) -> bool:
        return float(frame.mean()) < 4.0

    def text(self, frame=None) -> str:
        frame = self.frame() if frame is None else frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        # Xbox UI is mixed light-on-dark and dark-on-light; OCR both polarities.
        _, a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b = cv2.bitwise_not(a)
        out = [pytesseract.image_to_string(img, config="--psm 11") for img in (a, b)]
        return re.sub(r"\s+", " ", " ".join(out)).lower()

    def save(self, label: str, frame=None) -> None:
        if not self.save_dir:
            return
        frame = self.frame() if frame is None else frame
        self._shot += 1
        path = self.save_dir / f"{self._shot:03d}_{re.sub(r'[^a-z0-9]+', '_', label.lower())}.png"
        cv2.imwrite(str(path), frame)

    def release(self) -> None:
        self._halt()


def contains(haystack: str, needles: list[str], threshold: float = MATCH_THRESHOLD) -> str | None:
    """Return the first needle found in the OCR text (exact, then fuzzy), else None."""
    for n in needles:
        if n in haystack:
            return n
    words = haystack.split()
    for n in needles:
        k = len(n.split())
        for i in range(max(0, len(words) - k + 1)):
            chunk = " ".join(words[i:i + k])
            if difflib.SequenceMatcher(None, chunk, n).ratio() >= threshold:
                return n
    return None


# ----------------------------------------------------------------------------
# Live view (MJPEG over HTTP, works headless)
# ----------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xbox capture</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:1rem}
 img{max-width:100%;border:1px solid #444;display:block;margin-top:.75rem}
 select,button{font-size:1rem;padding:.3rem}
 #st{margin-left:1rem;opacity:.8}
</style>
<h3>Xbox capture</h3>
<label>Video input: <select id="dev"></select></label>
<span id="st"></span>
<img id="v" src="/stream">
<pre id="ocr" style="white-space:pre-wrap;font-size:.8rem;opacity:.7;max-width:960px;margin:.5rem 0 0"></pre>

<style>
 #remote{margin-top:1.5rem}
 #pad{display:grid;grid-template-columns:repeat(3,5rem);gap:.4rem;margin:.5rem 0}
 #pad button{padding:.8rem 0;font-size:1rem}
 #rstate{margin-left:.75rem;opacity:.8}
</style>
<div id="remote">
  <h3>Remote control</h3>
  <div id="pad">
    <button data-key="guide">Guide</button>
    <button data-key="up">&#9650;</button>
    <button data-key="esc">Back</button>
    <button data-key="left">&#9664;</button>
    <button data-key="enter">OK</button>
    <button data-key="right">&#9654;</button>
    <button data-key="tab">Tab</button>
    <button data-key="down">&#9660;</button>
    <button data-key="backspace">&#9003;</button>
  </div>
  <label><input type="checkbox" id="kbd"> Use my keyboard
    (arrows, Enter, Esc, Tab, Backspace, Space)</label>
  <span id="rstate"></span>
</div>

<style>
 #serial{margin-top:1.5rem;max-width:960px}
 #slog{background:#000;color:#9f9;height:12rem;overflow:auto;padding:.5rem;border:1px solid #444;
       white-space:pre-wrap;word-break:break-all;font-size:.85rem;margin:.5rem 0}
 #sin{width:60%;font-size:1rem;padding:.3rem}
 #quick button{margin:.15rem}
 .warn{color:#fc6}
</style>
<div id="serial">
  <h3>Serial (Web Serial)</h3>
  <div id="sunsupported" class="warn" hidden></div>
  <div id="sui">
    <label>Baud
      <select id="baud">
        <option>9600</option><option selected>115200</option><option>230400</option>
        <option>460800</option><option>921600</option>
      </select>
    </label>
    <button id="sconn">Connect</button>
    <button id="sdisc" disabled>Disconnect</button>
    <button id="sclear">Clear</button>
    <span id="sstate" style="margin-left:.5rem;opacity:.8">not connected</span>
    <pre id="slog"></pre>
    <form id="sform">
      <input id="sin" placeholder="type a command, e.g. down" autocomplete="off" disabled>
      <button id="ssend" disabled>Send</button>
    </form>
    <div id="quick"></div>
  </div>
</div>

<style>
 #learn{margin-top:1.5rem;max-width:960px}
 #learn label{display:block;margin:.4rem 0}
 #learn input[type=text],#learn textarea,#learn select{font-size:1rem;padding:.3rem;width:100%;
   max-width:30rem;box-sizing:border-box}
 #learn .hint{opacity:.75;font-size:.9rem;margin:.2rem 0 .6rem}
 #lform{border:1px solid #444;padding:.6rem .8rem;margin:.6rem 0;border-radius:4px}
 #lrules div{margin:.2rem 0}
 #lrules button{padding:.05rem .4rem}
</style>
<div id="learn">
  <h3>Learning mode</h3>
  <div class="hint">1) Go to a screen.&nbsp; 2) Click <b>Learn this screen</b>.&nbsp;
    3) Press the buttons that get past it with the remote above.&nbsp;
    4) Pick what the rule should do and click <b>Save rule</b>.</div>
  <button id="lcap">Learn this screen</button>
  <button id="lcancel" hidden>Cancel</button>
  <span id="lmsg" style="margin-left:.6rem"></span>
  <div id="lform" hidden>
    <div id="lnote" class="hint" hidden></div>
    <label>Rule name <input type="text" id="lname"></label>
    <label>Recognize the screen by these phrases (one per line)
      <textarea id="lphr" rows="3"></textarea></label>
    <div>Buttons pressed since you clicked Learn: <b id="lkeys">none yet</b></div>
    <label>When this screen appears:
      <select id="lact">
        <option value="keys">press the buttons I pressed</option>
        <option value="seek">pick a list item (press down until it is highlighted)</option>
        <option value="wait">just wait</option>
        <option value="pause">hand it over to me</option>
        <option value="done">this is the final screen (setup finished)</option>
      </select></label>
    <label id="lvisitrow">Only on visit number (leave empty = every time this screen appears)
      <input type="number" id="lvisit" min="1" step="1" placeholder="e.g. 2 for the second time"></label>
    <label id="lseekrow" hidden>Name of the list item
      <input type="text" id="lseek" placeholder="e.g. my WiFi network name"></label>
    <label id="lpauserow" hidden>Message to show
      <input type="text" id="lpause" placeholder="e.g. enter the WiFi password"></label>
    <button id="lsave">Save rule</button>
  </div>
  <h4>Rules in <code id="lpath"></code></h4>
  <div id="lrules"></div>
</div>

<script>
const dev = document.getElementById('dev'), st = document.getElementById('st'),
      v = document.getElementById('v');
async function loadDevices(){
  const d = await (await fetch('/devices')).json();
  dev.innerHTML = '';
  for (const x of d.devices){
    const o = document.createElement('option');
    o.value = x.index; o.textContent = x.name;
    if (x.index === d.current) o.selected = true;
    dev.appendChild(o);
  }
}
dev.onchange = async () => {
  const r = await fetch('/select?index=' + dev.value);
  if (!r.ok) alert(await r.text());
  v.src = '/stream?t=' + Date.now();
  loadDevices();
};
async function poll(){
  try{
    const s = await (await fetch('/status')).json();
    st.textContent = s.state + (s.detail ? ' - ' + s.detail : '');
    document.getElementById('ocr').textContent = s.ocr ? 'OCR: ' + s.ocr : '';
  }catch(e){}
  setTimeout(poll, 2000);
}
loadDevices(); poll();

// ---- Remote control (browser -> this server -> ESP32 over WiFi) -------------
const rstate = document.getElementById('rstate');
let ctrlToken = '';
try { ctrlToken = localStorage.getItem('xbox_token') || ''; } catch (e) {}

async function sendKey(key){
  const h = {'Content-Type': 'application/json'};
  if (ctrlToken) h['X-Token'] = ctrlToken;
  try {
    const r = await fetch('/key', {method: 'POST', headers: h, body: JSON.stringify({key})});
    if (r.status === 401) {
      const t = prompt('Control token:');
      if (t) {
        ctrlToken = t;
        try { localStorage.setItem('xbox_token', t); } catch (e) {}
        return sendKey(key);
      }
    }
    rstate.textContent = r.ok ? 'sent ' + key : 'error: ' + await r.text();
  } catch (e) {
    rstate.textContent = 'error: ' + e.message;
  }
}
document.querySelectorAll('#pad button').forEach(b => {
  b.onclick = () => sendKey(b.dataset.key);
});
const KEYMAP = {ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right',
                Enter:'enter', Escape:'esc', Tab:'tab', Backspace:'backspace', ' ':'space'};
document.addEventListener('keydown', e => {
  if (!document.getElementById('kbd').checked) return;
  const t = e.target;
  if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' ||
      (t.tagName === 'INPUT' && t.type !== 'checkbox')) return;
  const k = KEYMAP[e.key];
  if (!k) return;
  e.preventDefault();
  sendKey(k);
});

// ---- Learning mode ---------------------------------------------------------
const L = id => document.getElementById(id);
const NL = String.fromCharCode(10);
let formShown = false, lastRules = '';
async function learnApi(path, body){
  const h = {'Content-Type': 'application/json'};
  if (ctrlToken) h['X-Token'] = ctrlToken;
  const r = await fetch(path, {method: 'POST', headers: h, body: JSON.stringify(body || {})});
  if (r.status === 401) {
    const t = prompt('Control token:');
    if (t) {
      ctrlToken = t;
      try { localStorage.setItem('xbox_token', t); } catch (e) {}
      return learnApi(path, body);
    }
  }
  const text = await r.text();
  if (!r.ok) throw new Error(text);
  return text ? JSON.parse(text) : {};
}
function learnMsg(t, bad){ L('lmsg').textContent = t; L('lmsg').style.color = bad ? '#f88' : '#8f8'; }
function showForm(on){ L('lform').hidden = !on; L('lcancel').hidden = !on; formShown = on; }
function fillForm(s){
  L('lname').value = s.name || '';
  L('lphr').value = (s.phrases || []).join(NL);
  L('lvisit').value = s.visit || '';
  const note = L('lnote');
  if (s.existing) {
    note.textContent = 'This screen already has the rule "' + s.existing + '"'
      + (s.seen ? ' (seen ' + s.seen + 'x this run)' : '')
      + '. Saving with a visit number adds a different set of buttons for that visit to the same rule.';
    note.hidden = false;
  } else { note.hidden = true; }
  showForm(true);
}
function actChanged(){
  L('lseekrow').hidden = L('lact').value !== 'seek';
  L('lpauserow').hidden = L('lact').value !== 'pause';
  L('lvisitrow').hidden = L('lact').value !== 'keys';
}
L('lact').onchange = actChanged;
L('lcap').onclick = async () => {
  learnMsg('reading the screen...');
  try {
    fillForm(await learnApi('/learn/capture'));
    learnMsg('Now press the buttons for this screen on the remote.');
  } catch (e) { learnMsg(e.message, true); }
};
L('lcancel').onclick = async () => {
  try { await learnApi('/learn/cancel'); } catch (e) {}
  showForm(false); learnMsg('');
};
L('lsave').onclick = async () => {
  try {
    const r = await learnApi('/learn/save', {
      name: L('lname').value, phrases: L('lphr').value.split(NL),
      action: L('lact').value, seek_text: L('lseek').value, pause: L('lpause').value,
      visit: L('lvisit').value});
    learnMsg('Saved rule ' + r.saved + (r.visit ? ' (buttons for visit ' + r.visit + ')' : '')
             + (r.replaced ? ' (updated the existing rule)' : ''));
    showForm(false); lastRules = '';
  } catch (e) { learnMsg(e.message, true); }
};
function renderRules(rules){
  const j = JSON.stringify(rules);
  if (j === lastRules) return;
  lastRules = j;
  const box = L('lrules');
  box.textContent = '';
  for (const r of rules){
    const row = document.createElement('div');
    const x = document.createElement('button');
    x.textContent = '\u2715'; x.title = 'delete this rule';
    x.onclick = async () => {
      if (!confirm('Delete rule ' + r.name + '?')) return;
      try { await learnApi('/learn/delete', {name: r.name}); lastRules = ''; }
      catch (e) { learnMsg(e.message, true); }
    };
    const label = document.createElement('span');
    label.textContent = ' ' + r.name + ' \u2014 ' + r.summary
      + (r.seen ? '  [seen ' + r.seen + 'x this run]' : '');
    row.appendChild(x); row.appendChild(label);
    box.appendChild(row);
  }
}
async function learnPoll(){
  try {
    const resp = await fetch('/learn/state');
    if (!resp.ok) { L('learn').hidden = true; return; }     // learning not available here
    const s = await resp.json();
    L('lpath').textContent = s.path;
    if (s.active && !formShown) fillForm(s);
    if (!s.active && formShown) showForm(false);
    L('lkeys').textContent = s.keys.length
      ? s.keys.map(k => k[0] + (k[1] > 1 ? ' x' + k[1] : '')).join(', ') : 'none yet';
    renderRules(s.rules);
    if (s.error) learnMsg(s.error, true);
  } catch (e) {}
  setTimeout(learnPoll, 1500);
}
learnPoll();

// ---- Web Serial ------------------------------------------------------------
// The serial port belongs to the computer running this browser, not the server.
const $ = id => document.getElementById(id);
const slog = $('slog'), sin = $('sin'), ssend = $('ssend'),
      sconn = $('sconn'), sdisc = $('sdisc'), sstate = $('sstate');
let port = null, reader = null, writer = null, keepReading = false;

function slogAdd(t){
  slog.textContent += t;
  if (slog.textContent.length > 30000) slog.textContent = slog.textContent.slice(-20000);
  slog.scrollTop = slog.scrollHeight;
}
function setConnected(on){
  sconn.disabled = on; sdisc.disabled = !on; sin.disabled = !on; ssend.disabled = !on;
  $('baud').disabled = on;
  document.querySelectorAll('#quick button').forEach(b => b.disabled = !on);
  sstate.textContent = on ? 'connected' : 'not connected';
}
async function serialSend(text){
  if (!writer) return;
  try {
    await writer.write(new TextEncoder().encode(text + '\n'));
    slogAdd('> ' + text + '\n');
  } catch (e) { slogAdd('[send error] ' + e.message + '\n'); }
}
async function readLoop(){
  const dec = new TextDecoder();
  while (port && port.readable && keepReading){
    reader = port.readable.getReader();
    try {
      while (true){
        const {value, done} = await reader.read();
        if (done) break;
        slogAdd(dec.decode(value, {stream: true}));
      }
    } catch (e) {
      slogAdd('[read error] ' + e.message + '\n');
    } finally {
      reader.releaseLock();
    }
  }
}
async function serialClose(){
  keepReading = false;
  try { if (reader) await reader.cancel(); } catch (e) {}
  try { if (writer) { writer.releaseLock(); } } catch (e) {}
  try { if (port) await port.close(); } catch (e) {}
  port = reader = writer = null;
  setConnected(false);
}

if (!('serial' in navigator)) {
  const u = $('sunsupported');
  u.hidden = false;
  u.textContent = window.isSecureContext
    ? 'Web Serial is not supported in this browser. Use Chrome or Edge on desktop.'
    : 'Web Serial needs a secure context. Open this page via https:// or http://localhost '
      + '(for example: ssh -L 8080:localhost:8080 user@server), then reload.';
  $('sui').hidden = true;
} else {
  for (const k of ['up','down','left','right','enter','esc','tab','guide','status']) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = k; b.disabled = true;
    b.onclick = () => serialSend(k);
    $('quick').appendChild(b);
  }
  sconn.onclick = async () => {
    try {
      port = await navigator.serial.requestPort();
      await port.open({baudRate: parseInt($('baud').value, 10)});
      writer = port.writable.getWriter();
      keepReading = true;
      setConnected(true);
      readLoop();
    } catch (e) {
      slogAdd('[connect error] ' + e.message + '\n');
      port = null;
      setConnected(false);
    }
  };
  sdisc.onclick = serialClose;
  $('sclear').onclick = () => { slog.textContent = ''; };
  $('sform').onsubmit = e => {
    e.preventDefault();
    if (sin.value.trim()) { serialSend(sin.value.trim()); sin.value = ''; }
  };
  navigator.serial.addEventListener('disconnect', () => {
    slogAdd('[device disconnected]\n');
    serialClose();
  });
  setConnected(false);
}
</script>
"""


ALLOWED_KEYS = {"up", "down", "left", "right", "enter", "esc", "tab",
                "space", "backspace", "guide"}


def start_live_server(screen: Screen, devices: list[dict], port: int,
                      certfile: str | None = None, keyfile: str | None = None,
                      keys: "KeySender | None" = None,
                      control_token: str = "",
                      learner: "RuleLearner | None" = None) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/devices":
                body = json.dumps({"devices": devices, "current": screen.index})
                self._send(200, body.encode(), "application/json")
            elif url.path == "/status":
                self._send(200, json.dumps(STATUS).encode(), "application/json")
            elif url.path == "/select":
                try:
                    idx = int(parse_qs(url.query)["index"][0])
                    if idx not in [d["index"] for d in devices]:
                        raise ValueError(f"unknown device {idx}")
                    screen.switch(idx)
                    self._send(200, b"ok")
                except Exception as e:
                    self._send(400, f"Could not switch: {e}".encode())
            elif url.path == "/learn/state" and learner is not None:
                self._send(200, json.dumps(learner.state()).encode(), "application/json")
            elif url.path == "/stream":
                self._stream()
            else:
                self._send(404, b"not found")

        def do_POST(self):
            """Remote control (/key) and Learning mode (/learn/...)."""
            path = urlparse(self.path).path
            is_learn = path.startswith("/learn/")
            if (path == "/key" and keys is None) or (is_learn and learner is None) \
                    or not (path == "/key" or is_learn):
                self._send(404, b"not found")
                return
            if control_token and self.headers.get("X-Token", "") != control_token:
                self._send(401, b"unauthorized")
                return
            # JSON content type forces a CORS preflight, so other websites can't fire keys.
            if "application/json" not in self.headers.get("Content-Type", ""):
                self._send(415, b"send application/json")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 8192:
                    raise ValueError("body too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                if path == "/key":
                    key = str(body.get("key", ""))
                    if key not in ALLOWED_KEYS:
                        raise ValueError(f"unknown key: {key}")
                    keys.press(key, delay=0.05)
                    log.info("remote key: %s", key)
                    self._send(200, b'{"ok":true}', "application/json")
                elif path == "/learn/capture":
                    self._send(200, json.dumps(learner.capture()).encode(), "application/json")
                elif path == "/learn/save":
                    self._send(200, json.dumps(learner.save(body)).encode(), "application/json")
                elif path == "/learn/delete":
                    self._send(200, json.dumps(learner.delete(str(body.get("name", "")))).encode(),
                               "application/json")
                elif path == "/learn/cancel":
                    learner.cancel()
                    self._send(200, b'{"ok":true}', "application/json")
                else:
                    self._send(404, b"not found")
            except RuntimeError as e:        # ESP32 unreachable / no video frame
                self._send(502, str(e).encode())
            except Exception as e:
                self._send(400, str(e).encode())

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            interval = 1.0 / STREAM_FPS
            try:
                while True:
                    t0 = time.time()
                    frame = screen.frame()
                    label = STATUS["state"] + (f": {STATUS['detail']}" if STATUS["detail"] else "")
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
                    cv2.putText(frame, f"input {screen.index} | {label}"[:90], (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        data = jpg.tobytes()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(data)}\r\n\r\n".encode()
                                         + data + b"\r\n")
                    time.sleep(max(0, interval - (time.time() - t0)))
            except (BrokenPipeError, ConnectionResetError, RuntimeError):
                pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    scheme = "http"
    if certfile and keyfile:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Live view: %s://<this-machine-ip>:%d", scheme, port)
    return server


# ----------------------------------------------------------------------------
# Flow definition
# ----------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    keys: list[tuple[str, int]]            # (key, repeat count)
    expect: list[str]                      # text that must appear afterwards
    timeout: float = SCREEN_TIMEOUT
    retries: int = 1


# Navigation to Settings > System > Updates using keyboard.
# These key paths are guesses; verify by hand once, then edit here.
FLOW = [
    Step("open guide", [("guide", 1)], ["settings", "profile & system", "profile"]),
    Step("open settings", [("down", 1), ("enter", 1)], ["system", "general", "account"]),
    Step("open system", [("enter", 1)], ["updates", "update"]),
    Step("open updates", [("enter", 1)], UPDATE_AVAILABLE_TEXT + UP_TO_DATE_TEXT),
]


# ----------------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Setup mode: screen recognition with OpenCV (OCR text + template matching)
# ----------------------------------------------------------------------------
_template_cache: dict = {}


def load_template(name: str):
    if name not in _template_cache:
        path = TEMPLATE_DIR / f"{name}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
        if img is None:
            log.warning("Template not found: %s", path)
        _template_cache[name] = img
    return _template_cache[name]


def template_score(frame, name: str) -> float:
    """Best normalized match (0..1) of templates/<name>.png anywhere in the frame."""
    tpl = load_template(name)
    if tpl is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
        return 0.0
    score = float(cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).max())
    return score if score == score else 0.0      # NaN on perfectly flat images


def save_template(frame, name: str, roi: tuple[int, int, int, int] | None = None) -> Path:
    """Save the current frame (or an x,y,w,h crop of it) as templates/<name>.png."""
    if roi:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
        if frame.size == 0:
            raise ValueError("ROI is outside the frame")
    TEMPLATE_DIR.mkdir(exist_ok=True)
    path = TEMPLATE_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    return path


def normalize_keys(value, where: str = "keys") -> list:
    """Accept the forgiving ways of writing key presses in a rules file and return the
    canonical [[key, count], ...] form. All of these mean "down twice, then enter":
        [["down", 2], ["enter", 1]]      (canonical)
        [["down", 2], "enter"]           (bare name = one press)
        ["down", 2, "enter"]             (a number after a key = its count)
        ["down", "down", "enter"]
    A single string ("enter") or a single pair (["enter", 1]) also work."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list = []
    items = list(value)
    i = 0
    while i < len(items):
        item = items[i]
        if isinstance(item, (list, tuple)):
            out.extend(normalize_keys(item, where))
            i += 1
            continue
        if not isinstance(item, str):
            raise ValueError(f"{where}: expected a key name, got {item!r}")
        key = item.strip().lower()
        if key not in ALLOWED_KEYS:
            raise ValueError(f"{where}: unknown key {item!r} (allowed: {', '.join(sorted(ALLOWED_KEYS))})")
        count = 1
        if i + 1 < len(items) and isinstance(items[i + 1], int) and not isinstance(items[i + 1], bool):
            count = items[i + 1]
            i += 1
        if count < 1:
            raise ValueError(f"{where}: press count for {key!r} must be at least 1")
        out.append([key, count])
        i += 1
    return out


@dataclass
class Rule:
    """One known setup screen and what to do on it.

    A rule matches when at least one positive check passes:
      - `text`:     ANY of these phrases is on screen (OCR)
      - `all_text`: EVERY one of these phrases is on screen (stricter)
      - `template`: templates/<name>.png is found (OpenCV) with score >= `threshold`
    ...and none of the `not_text` phrases are on screen (a veto).
    Rules are checked in order; the first match wins.
    """
    name: str
    text: list = field(default_factory=list)
    all_text: list = field(default_factory=list)
    not_text: list = field(default_factory=list)
    template: str | None = None
    threshold: float = 0.85
    keys: list = field(default_factory=list)   # e.g. [["down", 2], ["enter", 1]]; empty = just wait
    pause: str | None = None                   # set to hand this screen to a human
    done: bool = False                         # setup is finished when this screen appears
    settle: float = 2.0                        # seconds to wait after pressing keys
    max_hits: int = 3                          # consecutive tries before calling it "stuck"
    # "Press a key until this item is highlighted" (e.g. pick your WiFi network in a list):
    seek_text: str | None = None               # phrase to find, e.g. the network name
    seek_key: str = "down"                     # key pressed to move through the list
    seek_max: int = 30                         # give up (hand over to you) after this many presses
    seek_then: list = field(default_factory=lambda: [["enter", 1]])  # keys once it is highlighted
    seek_contrast: float = 20.0                # how different the highlighted row must look
    seek_use_last_position: bool = True        # if OCR can't read the name once it's highlighted,
                                               # check the spot where it was last seen
    seek_wait_text: list = field(default_factory=lambda: [
        "continue", "it's all good", "connected to the internet"])
                                               # after selecting, wait for one of these phrases
                                               # (e.g. the Continue button) before moving on
    seek_wait_timeout: float = 90.0            # seconds to wait for it before handing over to you
    # Different keys on the Nth time this screen is seen (1 = first visit). A visit without an
    # entry here uses `keys`. In JSON the numbers are strings:
    #   "visit_keys": {"2": [["tab", 1], ["enter", 1]]}
    visit_keys: dict = field(default_factory=dict)

    def __post_init__(self):
        # Tolerate loosely written key lists in hand-edited JSON.
        self.keys = normalize_keys(self.keys, f"rule '{self.name}' keys")
        self.seek_then = normalize_keys(self.seek_then, f"rule '{self.name}' seek_then")
        fixed: dict = {}
        for k, v in (self.visit_keys or {}).items():
            try:
                n = int(k)
            except (TypeError, ValueError):
                raise ValueError(f"rule '{self.name}' visit_keys: {k!r} is not a visit number")
            if n < 1:
                raise ValueError(f"rule '{self.name}' visit_keys: visit numbers start at 1")
            fixed[n] = normalize_keys(v, f"rule '{self.name}' visit_keys[{n}]")
        self.visit_keys = fixed

    def keys_for_visit(self, visit: int) -> list:
        """Keys to press on the Nth visit: the per-visit override if there is one, else `keys`."""
        return self.visit_keys.get(visit, self.keys)


# Best-guess defaults for the first-time Xbox One setup. Run --ocr-test on each
# real screen and edit (or export with --write-default-rules and edit the JSON).
DEFAULT_RULES = [
    Rule("done", text=["you're all set", "my games & apps"], done=True),
    Rule("updating", text=["updating", "installing", "downloading", "% complete"],
         keys=[], settle=10, max_hits=999),
    Rule("language", all_text=["deutsch", "italiano"],     # first screen: language list
         keys=[["enter", 1]], settle=3),
    Rule("language_region", text=["select your language", "choose your language",
                                  "select your location", "choose your location"],
         pause="choose language / region"),
    Rule("wifi_select", text=["select your network", "choose a network", "wireless network"],
         pause="choose your WiFi network and enter the password"),
    Rule("sign_in", text=["sign in", "enter your email", "microsoft account"],
         pause="sign in to your Microsoft account"),
    Rule("terms", text=["license terms", "terms of use", "privacy statement",
                        "services agreement"],
         pause="review and accept the terms"),
    Rule("update_prompt", text=["console update", "update required", "install update",
                                "let's update"],
         keys=[["enter", 1]], settle=5),
    Rule("power_mode", text=["power mode", "instant-on", "energy-saving"],
         keys=[["enter", 1]]),
    Rule("welcome", text=["hello from xbox", "let's get started", "get started"],
         keys=[["enter", 1]]),
    Rule("network_ok", text=["you're connected", "connected to the network",
                             "it's all good", "connected to the internet"],
         keys=[["enter", 1]]),
    Rule("generic_next", text=["next", "continue", "got it"],
         not_text=["license", "terms", "privacy", "sign in", "password", "email"],
         keys=[["enter", 1]], max_hits=2),
]


def load_rules(path: str | None) -> list[Rule]:
    if not path:
        return DEFAULT_RULES
    with open(path, encoding="utf-8") as f:
        return [Rule(**d) for d in json.load(f)]


def evaluate_rule(rule: Rule, frame, text: str) -> tuple[bool, list[str]]:
    """Run one rule's checks against the screen. Returns (matched, explanation)."""
    why: list[str] = []
    positive = False
    if rule.text:
        hit = contains(text, [t.lower() for t in rule.text])
        why.append(f"text any-of: found {hit!r}" if hit else "text any-of: none found")
        positive = positive or bool(hit)
    if rule.all_text:
        missing = [t for t in rule.all_text if not contains(text, [t.lower()])]
        why.append("all_text: all found" if not missing else f"all_text: missing {missing}")
        positive = positive or not missing
    if rule.template:
        score = template_score(frame, rule.template)
        why.append(f"template '{rule.template}': {score:.2f} (need {rule.threshold})")
        positive = positive or score >= rule.threshold
    if positive and rule.not_text:
        veto = contains(text, [t.lower() for t in rule.not_text])
        if veto:
            why.append(f"vetoed by not_text {veto!r}")
            return False, why
    return positive, why


def match_rule(rules: list[Rule], frame, text: str) -> Rule | None:
    for r in rules:
        if evaluate_rule(r, frame, text)[0]:
            return r
    return None


# ----------------------------------------------------------------------------
# Seeking an item in a list (find it with OCR, tell if it is highlighted)
# ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("\u2019", "'"))


def find_text_box(frame, phrase: str, threshold: float = 0.8):
    """Locate `phrase` on screen with OCR. Returns (x, y, w, h) in frame pixels or None."""
    scale = 1.5
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    target = _norm(phrase)
    n = len(phrase.split())
    best = None
    # Plain grey + Otsu, both polarities: a highlighted row (text on a coloured tile)
    # often only reads in one of these.
    ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 41, 10)
    for img in (gray, cv2.bitwise_not(gray), a, cv2.bitwise_not(a), ada, cv2.bitwise_not(ada)):
        d = pytesseract.image_to_data(img, config="--psm 11",
                                      output_type=pytesseract.Output.DICT)
        words = [(d["text"][i], d["left"][i], d["top"][i], d["width"][i], d["height"][i])
                 for i in range(len(d["text"])) if d["text"][i].strip()]
        for size in range(max(1, n - 1), n + 2):
            for i in range(len(words) - size + 1):
                chunk = words[i:i + size]
                if max(w[2] for w in chunk) - min(w[2] for w in chunk) > max(w[4] for w in chunk):
                    continue                      # not on the same line
                score = difflib.SequenceMatcher(
                    None, target, _norm("".join(w[0] for w in chunk))).ratio()
                if score < threshold:
                    continue
                x0 = min(w[1] for w in chunk)
                y0 = min(w[2] for w in chunk)
                x1 = max(w[1] + w[3] for w in chunk)
                y1 = max(w[2] + w[4] for w in chunk)
                # Best text match first; among equals prefer the TIGHTEST box, because
                # OCR sometimes boxes a whole coloured tile instead of just the letters.
                key = (round(score, 1), -((x1 - x0) * (y1 - y0)))
                if best is None or key > best[0]:
                    best = (key, (x0, y0, x1 - x0, y1 - y0))
    if best is None:
        return None
    return tuple(int(v / scale) for v in best[1])


def box_contrast(frame, box) -> float:
    """How different the background behind `box` is from its surroundings. A highlighted
    list row stands out; an ordinary row looks like the rest of the list. Two tests, and
    the higher score wins:
      - vertical: differs from BOTH the area just above and just below the row
      - sideways: differs from BOTH screen edges on the same line
    (Median colour is used so thin text strokes don't count.)"""
    x, y, w, h = box
    H, W = frame.shape[:2]
    x0, x1 = max(0, x), min(W, x + w)
    t0, t1 = max(0, y), min(H, y + h)
    if x1 - x0 < 3 or t1 - t0 < 3:
        return 0.0

    def med(ya, yb, xa, xb):
        ya, yb, xa, xb = max(0, ya), min(H, yb), max(0, xa), min(W, xb)
        if yb - ya < 3 or xb - xa < 3:
            return None
        return np.median(frame[ya:yb, xa:xb].reshape(-1, 3), axis=0)

    here = med(t0, t1, x0, x1)
    strip = max(8, W // 30)

    def score(others):
        diffs = [float(np.abs(here - o).mean()) for o in others if o is not None]
        return min(diffs) if diffs else 0.0

    vertical = score([med(y - h, y, x0, x1), med(y + h, y + 2 * h, x0, x1)])
    sideways = score([med(t0, t1, 0, strip), med(t0, t1, W - strip, W)])
    return max(vertical, sideways)


def seek_status(frame, rule: "Rule") -> tuple[bool, bool, float]:
    """(found on screen, looks highlighted, contrast score) for rule.seek_text."""
    box = find_text_box(frame, rule.seek_text)
    if box is None:
        return False, False, 0.0
    c = box_contrast(frame, box)
    return True, c >= rule.seek_contrast, c


# ----------------------------------------------------------------------------
# Calibration: learn setup rules by watching you go through setup once
# ----------------------------------------------------------------------------
def _words(text: str) -> set:
    return set(re.findall(r"[a-z']{3,}", text))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _phrases(text: str) -> set:
    """2- and 3-word phrases made only of plain words (no digits/symbols)."""
    toks = re.findall(r"[a-z0-9'&%]+", text)
    out = set()
    for n in (3, 2):
        for i in range(len(toks) - n + 1):
            chunk = toks[i:i + n]
            if all(re.fullmatch(r"[a-z']{3,}", t) for t in chunk):
                out.add(" ".join(chunk))
    return out


def _group_keys(seq: list[str]) -> list[list]:
    out: list[list] = []
    for k in seq:
        if out and out[-1][0] == k:
            out[-1][1] += 1
        else:
            out.append([k, 1])
    return out


class Calibrator:
    """Watches the screen while you click through setup with the web remote.

    Every time a new screen appears it records the OCR text and a screenshot,
    and attributes the keys you pressed since the previous screen to that
    previous screen. Rules are written to a JSON file after every new screen.
    """

    def __init__(self, screen: Screen, keys: KeySender, out_path: str, poll: float = 1.5):
        self.screen, self.keys, self.out_path, self.poll = screen, keys, out_path, poll
        self.records: list[dict] = []

    def _distinctive(self) -> list[list[str]]:
        sets = [_phrases(r["text"]) for r in self.records]
        result = []
        for i, mine in enumerate(sets):
            others = set().union(*(s for j, s in enumerate(sets) if j != i))
            uniq = sorted(mine - others, key=lambda p: (-len(p.split()), -len(p)))
            picked: list[str] = []
            for p in uniq:
                if any(p in q for q in picked):
                    continue
                picked.append(p)
                if len(picked) == 3:
                    break
            result.append(picked)
        return result

    def write(self) -> None:
        from dataclasses import asdict
        if not getattr(self, "_backed_up", False):
            self._backed_up = True
            if Path(self.out_path).exists():
                shutil.copy2(self.out_path, f"{self.out_path}.bak")
                log.warning("%s already exists and will be overwritten; the old copy is in %s.bak",
                            self.out_path, self.out_path)
        phrases = self._distinctive()
        rules = []
        for i, (rec, ph) in enumerate(zip(self.records, phrases)):
            last = i == len(self.records) - 1
            slug = re.sub(r"[^a-z0-9]+", "_", ph[0])[:24].strip("_") if ph else "screen"
            keys = [] if last else _group_keys(rec["keys"])
            waiting = not last and not keys          # screen moved on by itself
            rules.append(Rule(name=f"{i + 1:02d}_{slug}", text=ph, keys=keys, done=last,
                              settle=5.0 if waiting else 2.0,
                              max_hits=999 if waiting else 3))
            if not ph:
                log.warning("Screen %d has no distinctive text; add a template or text by hand.", i + 1)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rules], f, indent=2)

    def _add(self, frame, text: str, words: set, boundary: float) -> None:
        if self.records:
            prev = self.records[-1]
            prev["keys"] = self.keys.keys_between(prev["t"], boundary)
        self.records.append({"t": boundary, "text": text, "words": words, "keys": []})
        self.screen.save(f"calib_{len(self.records):02d}", frame)
        self.write()
        log.info("Screen %d recorded. OCR: %.300s", len(self.records), text)
        log.info("Saved %s (%d screens). Keep going; Ctrl+C when setup is finished.",
                 self.out_path, len(self.records))

    def run(self) -> None:
        log.info("Calibrating: go through setup using the remote on the live page. "
                 "Press Ctrl+C here when you reach the final screen.")
        set_status("calibrating", "go through setup with the remote")
        prev_words = None
        changed_at = None
        while True:
            frame = self.screen.frame()
            if self.screen.is_black(frame):
                time.sleep(self.poll)
                continue
            text = self.screen.text(frame).replace("’", "'")
            set_ocr(text)
            words = _words(text)

            current = self.records[-1]["words"] if self.records else None
            if current is not None and _jaccard(words, current) >= 0.7:
                changed_at = None                      # still on the recorded screen
            elif changed_at is None:
                changed_at = time.time()               # first sign of a different screen

            stable = prev_words is not None and _jaccard(words, prev_words) >= 0.8
            prev_words = words
            known = any(_jaccard(words, r["words"]) >= 0.7 for r in self.records)
            if stable and len(words) >= 2 and not known:
                # keys pressed up to ~1s before the change started caused it
                boundary = (changed_at or time.time()) - 1.0
                self._add(frame, text, words, boundary)
                changed_at = None
            time.sleep(self.poll)


class RuleLearner:
    """Backs the Learning mode panel on the live page.

    "Learn this screen" reads the screen and suggests phrases to recognize it by, then
    records every button you press on the web remote. "Save rule" adds (or replaces,
    by name) one rule in the rules JSON file. The rest of the file is left as it is,
    so hand edits and earlier rules are kept, and a .bak copy is made first.
    New rules go at the END of the file; the first matching rule wins, so move one
    up by hand if an earlier rule shadows it.
    """

    def __init__(self, screen: "Screen", keys: "KeySender", path: str):
        self.screen, self.keys, self.path = screen, keys, str(path)
        self.pending: dict | None = None
        self._lock = threading.Lock()
        self._backed_up = False

    # -- file access ---------------------------------------------------------
    def _read(self) -> list:
        p = Path(self.path)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{self.path} is not valid JSON ({e}). Fix it first; "
                             "nothing was changed.")
        if not isinstance(data, list):
            raise ValueError(f"{self.path} must contain a JSON list of rules")
        return data

    def _write(self, data: list) -> None:
        p = Path(self.path)
        if p.exists() and not self._backed_up:
            shutil.copy2(p, str(p) + ".bak")
            self._backed_up = True
        tmp = Path(str(p) + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    @staticmethod
    def _summary(raw: dict) -> str:
        try:
            return describe_action(Rule(**raw))
        except Exception:
            return "(check this rule)"

    # -- suggestions -----------------------------------------------------------
    @staticmethod
    def _suggest(text: str, existing: list) -> list[str]:
        used: set = set()
        for r in existing:
            if isinstance(r, dict):
                for field_name in ("text", "all_text"):
                    used.update(str(t).lower() for t in r.get(field_name) or [])
        picked: list[str] = []
        for ph in sorted(_phrases(text) - used, key=lambda x: (-len(x.split()), -len(x), x)):
            if any(ph in q or q in ph for q in picked):
                continue
            picked.append(ph)
            if len(picked) == 3:
                break
        if not picked:      # no clean multi-word phrases: fall back to the longest words
            picked = sorted(_words(text) - used, key=lambda w: (-len(w), w))[:3]
        return picked

    @staticmethod
    def _auto_name(phrases: list[str], existing: list) -> str:
        names = {r.get("name") for r in existing if isinstance(r, dict)}
        slug = re.sub(r"[^a-z0-9]+", "_", phrases[0])[:24].strip("_") if phrases else "screen"
        i = len(existing) + 1
        while f"{i:02d}_{slug}" in names:
            i += 1
        return f"{i:02d}_{slug}"

    # -- actions ---------------------------------------------------------------
    def capture(self) -> dict:
        t0 = time.time()                       # keys pressed from now on belong to this screen
        frame = self.screen.frame()
        text = self.screen.text(frame).replace("\u2019", "'")
        set_ocr(text)
        with self._lock:
            existing = self._read()
            parsed = []
            for r in existing:
                try:
                    parsed.append(Rule(**r))
                except Exception:
                    pass                              # a broken rule shouldn't block learning
            match = match_rule(parsed, frame, text)
            if match is not None:
                # This screen already has a rule: reuse its name so a visit number adds an
                # override to it, and suggest the next visit that has no override yet.
                name = match.name
                phrases = list(match.text) or self._suggest(text, existing)
                visit = (max(match.visit_keys) + 1 if match.visit_keys
                         else (2 if match.keys else ""))
            else:
                phrases = self._suggest(text, existing)
                name = self._auto_name(phrases, existing)
                visit = ""
            self.pending = {"t0": t0, "phrases": phrases, "name": name,
                            "existing": match.name if match else "", "visit": visit}
        self.screen.save(f"learn_{name}", frame)
        set_status("learning", "press the buttons for this screen")
        log.info("Learning screen '%s'%s. OCR: %.300s", name,
                 " (already has a rule)" if match else "", text)
        return {"name": name, "phrases": phrases, "ocr": text[:400],
                "existing": match.name if match else "", "visit": visit,
                "seen": VISITS.get(name, 0)}

    def cancel(self) -> None:
        with self._lock:
            self.pending = None
        set_status("idle", "")

    def state(self) -> dict:
        with self._lock:
            try:
                raw, err = self._read(), None
            except ValueError as e:
                raw, err = [], str(e)
            p = self.pending
            return {
                "active": p is not None,
                "name": p["name"] if p else "",
                "phrases": p["phrases"] if p else [],
                "existing": p.get("existing", "") if p else "",
                "visit": p.get("visit", "") if p else "",
                "seen": VISITS.get(p.get("existing", ""), 0) if p else 0,
                "keys": _group_keys(self.keys.keys_between(p["t0"], time.time())) if p else [],
                "rules": [{"name": str(r.get("name", "?")), "summary": self._summary(r),
                           "seen": VISITS.get(str(r.get("name", "?")), 0)}
                          for r in raw if isinstance(r, dict)],
                "path": self.path,
                "error": err,
            }

    def save(self, spec: dict) -> dict:
        with self._lock:
            p = self.pending
            if p is None:
                raise ValueError("Click 'Learn this screen' first.")
            phrases = [str(x).strip().lower().replace("\u2019", "'")
                       for x in spec.get("phrases") or [] if str(x).strip()]
            if not phrases:
                raise ValueError("Add at least one phrase to recognize the screen by.")
            name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(spec.get("name") or "")).strip("_") or p["name"]
            rule: dict = {"name": name, "text": phrases}
            action = spec.get("action")
            visit_n = None                                # set when saving keys for one visit only
            if action == "keys":
                pressed = _group_keys(self.keys.keys_between(p["t0"], time.time()))
                if not pressed:
                    raise ValueError("No buttons have been pressed since you clicked Learn this "
                                     "screen. Press them on the remote first, or pick another action.")
                v = str(spec.get("visit") or "").strip()
                if v:
                    try:
                        visit_n = int(v)
                    except ValueError:
                        visit_n = 0
                    if visit_n < 1:
                        raise ValueError("The visit number must be a whole number, 1 or higher "
                                         "(or leave it empty to use these buttons every time).")
                    rule["visit_keys"] = {str(visit_n): pressed}
                else:
                    rule["keys"] = pressed
            elif action == "seek":
                item = str(spec.get("seek_text") or "").strip()
                if not item:
                    raise ValueError("Type the name of the list item to pick.")
                rule["seek_text"] = item
            elif action == "wait":
                rule.update(settle=5.0, max_hits=999)
            elif action == "pause":
                rule["pause"] = str(spec.get("pause") or "").strip() or "do this screen yourself"
            elif action == "done":
                rule["done"] = True
            else:
                raise ValueError("Choose what the rule should do.")
            Rule(**rule)                                  # validate before touching the file
            raw = self._read()
            replaced = False
            for i, r in enumerate(raw):
                if isinstance(r, dict) and r.get("name") == name:
                    if visit_n is not None:
                        # Add this visit's keys to the existing rule; leave everything else alone.
                        merged = dict(r)
                        vk = dict(merged.get("visit_keys") or {})
                        vk[str(visit_n)] = rule["visit_keys"][str(visit_n)]
                        merged["visit_keys"] = vk
                        Rule(**merged)                # validate
                        rule = merged
                    elif action == "keys" and r.get("visit_keys"):
                        rule["visit_keys"] = r["visit_keys"]   # keep per-visit overrides
                    raw[i], replaced = rule, True
                    break
            if not replaced:
                raw.append(rule)
            self._write(raw)
            self.pending = None
        set_status("idle", "")
        log.info("Learned rule '%s' -> %s (%s)", name, self.path, self._summary(rule))
        return {"saved": name, "replaced": replaced, "visit": visit_n}

    def delete(self, name: str) -> dict:
        with self._lock:
            raw = self._read()
            keep = [r for r in raw if not (isinstance(r, dict) and r.get("name") == name)]
            if len(keep) == len(raw):
                raise ValueError(f"No rule named '{name}'.")
            self._write(keep)
        log.info("Deleted rule '%s' from %s", name, self.path)
        return {"deleted": name}


@dataclass
class Updater:
    keys: KeySender
    screen: Screen
    history: list[str] = field(default_factory=list)
    seek_done: set = field(default_factory=set)   # seek rules that already picked their item
    visit_similarity: float = VISIT_SIMILARITY    # see VISIT_SIMILARITY

    def wait_for(self, expect: list[str], timeout: float, label: str) -> str | None:
        end = time.time() + timeout
        last = ""
        while time.time() < end:
            frame = self.screen.frame()
            if self.screen.is_black(frame):
                time.sleep(1)
                continue
            last = self.screen.text(frame)
            hit = contains(last, expect)
            if hit:
                self.screen.save(f"{label}_ok", frame)
                return hit
            time.sleep(1)
        log.warning("Timed out waiting for %s. Last OCR: %.200s", expect, last)
        self.screen.save(f"{label}_timeout")
        return None

    def back_to_home(self) -> None:
        for _ in range(4):
            self.keys.press("esc")
        self.keys.press("guide")
        self.keys.press("esc")

    def run_flow(self) -> bool:
        for step in FLOW:
            ok = False
            for attempt in range(step.retries + 1):
                log.info("Step: %s (attempt %d)", step.name, attempt + 1)
                set_status("navigating", f"{step.name} (attempt {attempt + 1})")
                for key, n in step.keys:
                    self.keys.press(key, n)
                if self.wait_for(step.expect, step.timeout, step.name):
                    ok = True
                    break
                self.back_to_home()
            if not ok:
                log.error("Step failed: %s", step.name)
                set_status("failed", f"step: {step.name}")
                return False
        return True

    def check_and_update(self) -> str:
        """Returns 'updated', 'up_to_date', or 'failed'."""
        set_status("checking", "reading update screen")
        text = self.screen.text()
        if contains(text, UP_TO_DATE_TEXT):
            log.info("Console is already up to date.")
            set_status("done", "already up to date")
            return "up_to_date"
        if not contains(text, UPDATE_AVAILABLE_TEXT):
            log.error("Could not tell whether an update is available. OCR: %.300s", text)
            self.screen.save("update_state_unknown")
            set_status("failed", "update state unknown")
            return "failed"

        log.info("Update available - starting install.")
        self.keys.press("enter")
        return self.monitor_update()

    def wait_for_human(self, reason: str, rule: Rule | None, rules: list[Rule],
                       timeout: float = PAUSE_TIMEOUT) -> None:
        """Hold until the screen changes (you used the web remote) or time runs out."""
        log.warning("Needs you: %s", reason)
        set_status("needs you", reason)
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(2)
            frame = self.screen.frame()
            if self.screen.is_black(frame):
                continue
            now = match_rule(rules, frame, self.screen.text(frame).replace("’", "'"))
            if rule is None:
                if now is not None:
                    return
            elif now is None or now.name != rule.name:
                return
        log.warning("Gave up waiting for you after %d s", timeout)

    def seek(self, rule: Rule) -> str:
        """Press rule.seek_key until rule.seek_text is highlighted, then press
        rule.seek_then, then wait for rule.seek_wait_text (e.g. the Continue button).
        Returns "ok", "not_found" or "wait_timeout".

        OCR often can't read a name once its row is highlighted (text on a coloured
        tile), so once the name has been seen we also check its last known position.
        A debug image of every step is saved to the log folder."""
        last_box = None
        for i in range(rule.seek_max + 1):
            frame = self.screen.frame()
            box = find_text_box(frame, rule.seek_text)
            how = "read by OCR"
            if box is not None:
                last_box = box
            elif rule.seek_use_last_position and last_box is not None:
                box, how = last_box, "OCR can't read it - using last seen position"
            contrast = box_contrast(frame, box) if box else 0.0
            hi = box is not None and contrast >= rule.seek_contrast
            log.info("Seeking '%s': %s (contrast %.0f, need %g, press %d/%d)", rule.seek_text,
                     ("HIGHLIGHTED - " + how) if hi else ("not highlighted - " + how if box else "not visible"),
                     contrast, rule.seek_contrast, i, rule.seek_max)
            set_status("setup", f"{rule.name}: looking for '{rule.seek_text}'")
            dbg = frame.copy()
            if box:
                x, y, w, h = box
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0) if hi else (0, 0, 255), 3)
            cv2.putText(dbg, f"seek {rule.seek_text!r}: contrast {contrast:.0f} / {rule.seek_contrast:g}"
                        f" {'HIGHLIGHTED' if hi else ''}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2, cv2.LINE_AA)
            self.screen.save(f"seek_{rule.name}_{i:02d}", dbg)
            if hi:
                for key, n in rule.seek_then:
                    self.keys.press(key, n)
                self.seek_done.add(rule.name)     # never seek on this rule again this run
                return "ok" if self._wait_after_seek(rule) else "wait_timeout"
            if i < rule.seek_max:
                self.keys.press(rule.seek_key)
                time.sleep(0.3)
        return "not_found"

    def _wait_after_seek(self, rule: Rule) -> bool:
        """After picking the item, the screen may sit on a 'connecting' view that still
        shows the name. Wait for the Continue button (rule.seek_wait_text) to appear."""
        if not rule.seek_wait_text:
            return True
        want = [w.lower() for w in rule.seek_wait_text]
        started = time.time()
        time.sleep(1)                              # let the screen change after the key press
        end = started + rule.seek_wait_timeout
        text, frame = "", None
        while time.time() < end:
            frame = self.screen.frame()
            text = self.screen.text(frame).replace("\u2019", "'")
            set_ocr(text)
            set_status("setup", f"{rule.name}: waiting for {rule.seek_wait_text}")
            if contains(text, want):
                log.info("Saw %s after %.0f s - continuing.", rule.seek_wait_text,
                         time.time() - started)
                self.screen.save(f"seek_{rule.name}_continue", frame)
                return True
            time.sleep(0.3)                        # reading the screen already takes a second or so
        # Record what the screen actually showed, so you can see why it never matched.
        log.warning("Never saw %s after selecting '%s'. Last OCR: %.400s",
                    rule.seek_wait_text, rule.seek_text, text)
        if frame is not None:
            self.screen.save(f"seek_{rule.name}_wait_timeout", frame)
        return False

    def run_setup(self, rules: list[Rule], max_steps: int = 300) -> bool:
        """Recognize each setup screen and act on it, handing off to you when needed.
        Counts how many times each screen is seen (see VISITS) and logs a summary at the end."""
        VISITS.clear()
        try:
            return self._run_setup(rules, max_steps)
        finally:
            log_visit_summary()

    def _run_setup(self, rules: list[Rule], max_steps: int) -> bool:
        last, streak = None, 0
        prev_words: set | None = None            # words on the last matched frame
        unmatched_since = None
        for _ in range(max_steps):
            frame = self.screen.frame()
            if self.screen.is_black(frame):
                set_status("setup", "black screen (waiting)")
                time.sleep(3)
                continue
            text = self.screen.text(frame).replace("’", "'")
            set_ocr(text)
            # A seek rule that already picked its item is skipped from then on, so a
            # "connecting" screen that still shows the name can't trigger it again.
            active = [r for r in rules if r.name not in self.seek_done]
            rule = match_rule(active, frame, text)

            if rule is None:
                if unmatched_since is None:
                    unmatched_since = time.time()
                    log.info("Unrecognized screen. OCR: %.400s", text)
                set_status("setup", "unrecognized screen")
                if time.time() - unmatched_since > UNMATCHED_TIMEOUT:
                    log.warning("Unrecognized screen. OCR: %.300s", text)
                    self.screen.save("setup_unrecognized", frame)
                    self.wait_for_human("unrecognized screen - use the remote, "
                                        "or add a rule for it", None, rules)
                    unmatched_since = None
                    last, streak = None, 0
                time.sleep(2)
                continue

            unmatched_since = None
            words = _words(text)
            sim = _jaccard(words, prev_words) if prev_words is not None else None
            if rule.name == last and sim is not None and sim >= self.visit_similarity:
                streak += 1                       # same screen still up: another try, same visit
            else:
                # First time here, back after another screen, or the same rule now showing
                # visibly different content (e.g. a second sign-in page): a new visit.
                streak = 1
                VISITS[rule.name] = VISITS.get(rule.name, 0) + 1
                extra = (f" (same rule, but the screen changed: text similarity {sim:.2f} < "
                         f"{self.visit_similarity:g})" if rule.name == last and sim is not None else "")
                log.info("Screen '%s' seen %d time(s) so far%s", rule.name, VISITS[rule.name], extra)
                save_visit_counts()
                self.screen.save(f"setup_{rule.name}_visit{VISITS[rule.name]}", frame)
            last = rule.name
            prev_words = words
            visit = VISITS[rule.name]

            if rule.done:
                # Re-check a few seconds apart so one misread frame can't end setup early.
                held = True
                for _ in range(2):
                    time.sleep(2)
                    f2 = self.screen.frame()
                    t2 = self.screen.text(f2).replace("’", "'")
                    if match_rule(active, f2, t2) is not rule:
                        held = False
                        break
                if not held:
                    log.info("'%s' did not hold on re-check; continuing.", rule.name)
                    last, streak = None, 0
                    continue
                why = "; ".join(evaluate_rule(rule, frame, text)[1])
                log.info("Setup finished (%s) [%s]", rule.name, why)
                set_status("done", "setup finished")
                return True
            if rule.pause or streak > rule.max_hits:
                self.wait_for_human(rule.pause or f"stuck on '{rule.name}'", rule, rules)
                last, streak = None, 0
                continue

            why = "; ".join(evaluate_rule(rule, frame, text)[1])
            log.info("Setup screen: %s (visit %d, try %d) [%s]", rule.name, visit, streak, why)
            set_status("setup", f"{rule.name} (visit {visit})")
            if rule.seek_text:
                result = self.seek(rule)
                if result != "ok":
                    reason = (f"couldn't find '{rule.seek_text}' - pick it yourself"
                              if result == "not_found" else
                              f"picked '{rule.seek_text}' but never saw {rule.seek_wait_text} - "
                              "check the connection")
                    self.wait_for_human(reason, rule, rules)
                    last, streak = None, 0
                    continue
                if not rule.seek_wait_text:        # otherwise we already waited for the button
                    time.sleep(rule.settle)
                last, streak = None, 0
                continue
            to_press = rule.keys_for_visit(visit)
            if rule.visit_keys:
                log.info("Visit %d of '%s': pressing %s%s", visit, rule.name,
                         ", ".join(f"{k} x{n}" for k, n in to_press) or "nothing",
                         "" if visit in rule.visit_keys else " (default keys, no override for this visit)")
            for key, n in to_press:
                self.keys.press(key, n)
            time.sleep(rule.settle)

        log.error("Setup step limit reached")
        set_status("failed", "setup step limit")
        return False

    def monitor_update(self) -> str:
        start = time.time()
        seen_progress = False
        last_log = 0.0
        set_status("updating", "started")
        while time.time() - start < UPDATE_TIMEOUT:
            frame = self.screen.frame()
            if self.screen.is_black(frame):
                # Console is probably rebooting into the offline updater / restarting.
                seen_progress = True
                set_status("updating", "black screen (rebooting?)")
                time.sleep(5)
                continue
            text = self.screen.text(frame)

            if contains(text, ERROR_TEXT):
                log.error("Error screen detected: %.200s", text)
                self.screen.save("update_error", frame)
                set_status("failed", "error screen detected")
                return "failed"
            if contains(text, UPDATING_TEXT) or contains(text, RESTART_TEXT):
                seen_progress = True
                set_status("updating", f"{int(time.time() - start)}s elapsed")
                if time.time() - last_log > 60:
                    log.info("Update in progress...")
                    self.screen.save("progress", frame)
                    last_log = time.time()
            elif seen_progress and (contains(text, UP_TO_DATE_TEXT) or "home" in text):
                log.info("Update finished.")
                self.screen.save("update_done", frame)
                set_status("done", "update finished")
                return "updated"
            time.sleep(5)
        log.error("Update timed out after %d s", UPDATE_TIMEOUT)
        set_status("failed", "timed out")
        return "failed"



# ----------------------------------------------------------------------------
# Rule checking
# ----------------------------------------------------------------------------
def describe_action(rule: Rule) -> str:
    base = _describe_base(rule)
    if rule.visit_keys and not (rule.done or rule.pause or rule.seek_text):
        parts = [f"visit {n}: " + (", ".join(f"{k} x{c}" for k, c in ks) or "nothing")
                 for n, ks in sorted(rule.visit_keys.items())]
        base += "; " + "; ".join(parts)
        if not rule.keys:
            base = base.replace(f"wait {rule.settle:g}s", "other visits: wait " + f"{rule.settle:g}s", 1)
    return base


def _describe_base(rule: Rule) -> str:
    if rule.done:
        return "finish setup"
    if rule.pause:
        return f"hand over to you ({rule.pause})"
    if rule.seek_text:
        then = ", ".join(f"{k} x{n}" for k, n in rule.seek_then)
        wait = f", then wait for {rule.seek_wait_text}" if rule.seek_wait_text else ""
        return (f"press {rule.seek_key} until '{rule.seek_text}' is highlighted "
                f"(max {rule.seek_max}), then {then}{wait}")
    if rule.keys:
        return "press " + ", ".join(f"{k} x{n}" for k, n in rule.keys)
    return f"wait {rule.settle:g}s"


def check_rules_once(screen: "Screen", rules: list[Rule]) -> tuple[str, tuple]:
    """Grab the current frame and evaluate every rule.
    Returns (printable report, signature). The signature only changes when the set
    of matching rules changes, so a loop can print only when something new shows up."""
    frame = screen.frame()
    text = screen.text(frame).replace("\u2019", "'")
    set_ocr(text)
    lines = [f"OCR text: {text[:600]}", ""]
    matched = []
    for r in rules:
        ok, why = evaluate_rule(r, frame, text)
        if ok:
            matched.append(r.name)
        lines.append(f"[{'MATCH' if ok else '  -  '}] {r.name}: " + "; ".join(why or ["no checks defined"]))
        if r.seek_text and ok:
            found, hi, c = seek_status(frame, r)
            lines.append(f"        seek '{r.seek_text}': "
                         + ("not on screen" if not found else
                            f"on screen, {'HIGHLIGHTED' if hi else 'not highlighted'} "
                            f"(contrast {c:.0f}, need {r.seek_contrast:g})"))
    winner = match_rule(rules, frame, text)
    if winner is None:
        lines.append("\nNo rule matches this screen.")
        set_status("check-rules", "no rule matches")
    else:
        lines.append(f"\nFirst match -> {winner.name}: {describe_action(winner)}")
        set_status("check-rules", f"match: {winner.name}")
    return "\n".join(lines), tuple(matched)


def run_check_rules(screen: "Screen", rules: list[Rule], interval: float) -> None:
    """Keep checking the screen until Ctrl+C. Prints a full report whenever the set
    of matching rules changes, and a one-line heartbeat in between."""
    log.info("Checking rules every %.1fs. Ctrl+C to stop.", interval)
    last_sig = None
    n = 0
    try:
        while True:
            try:
                report, sig = check_rules_once(screen, rules)
            except RuntimeError as e:          # e.g. a momentary capture hiccup
                log.warning("%s - retrying", e)
                time.sleep(interval)
                continue
            n += 1
            if sig != last_sig:
                print(f"\n===== {time.strftime('%H:%M:%S')} screen changed =====\n{report}", flush=True)
                last_sig = sig
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] same matches "
                      f"({', '.join(sig) or 'none'}) - check #{n}   ", end="", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped checking rules.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Xbox One auto-updater")
    p.add_argument("--esp-url", default=ESP_URL)
    p.add_argument("--device", type=int, default=None,
                   help="video input index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true", help="list video inputs and exit")
    p.add_argument("--live", action="store_true", help="serve live video in a browser")
    p.add_argument("--view-only", action="store_true",
                   help="serve live video only; don't run the updater")
    p.add_argument("--port", type=int, default=LIVE_PORT)
    p.add_argument("--cert", help="TLS certificate (enables https, needed for Web Serial off-localhost)")
    p.add_argument("--key", help="TLS private key for --cert")
    p.add_argument("--token", default="",
                   help="require this token (X-Token) for remote control from the web page")
    p.add_argument("--esp-token", default="",
                   help="token to send to the ESP32 (must match AUTH_TOKEN in the firmware)")
    p.add_argument("--setup", action="store_true",
                   help="get through first-time Xbox setup using screen recognition")
    p.add_argument("--then-update", action="store_true",
                   help="after --setup, run the system update check")
    p.add_argument("--rules", metavar="FILE", help="setup rules JSON (see --write-default-rules)")
    p.add_argument("--write-default-rules", metavar="FILE",
                   help="write the built-in setup rules to FILE for editing, then exit")
    p.add_argument("--setup-max-steps", type=int, default=300)
    p.add_argument("--visit-similarity", type=float, default=VISIT_SIMILARITY, metavar="0-1",
                   help="how alike two readings of the same rule's screen must be to count as the "
                        "same visit (default %(default)s). Raise it if a second page is counted as "
                        "the same visit; lower it if one page is counted as several.")
    p.add_argument("--calibrate", action="store_true",
                   help="learn setup rules by watching you go through setup with the web remote")
    p.add_argument("--calibrate-out", default="calibrated_rules.json", metavar="FILE")
    p.add_argument("--check-rules", action="store_true",
                   help="keep comparing the screen to every setup rule and print the result "
                        "whenever it changes (Ctrl+C to stop)")
    p.add_argument("--check-once", action="store_true",
                   help="with --check-rules: check a single time and exit")
    p.add_argument("--check-interval", type=float, default=2.0, metavar="SEC",
                   help="seconds between checks for --check-rules (default 2)")
    p.add_argument("--save-template", metavar="NAME",
                   help="save the current frame (or --roi crop) as templates/NAME.png and exit")
    p.add_argument("--roi", metavar="X,Y,W,H", help="crop for --save-template (pixels)")
    p.add_argument("--dry-run", action="store_true", help="print keys instead of sending")
    p.add_argument("--ocr-test", action="store_true", help="print OCR text of current frame and exit")
    p.add_argument("--snapshot", metavar="FILE", help="save one frame and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.write_default_rules:
        from dataclasses import asdict
        with open(args.write_default_rules, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in DEFAULT_RULES], f, indent=2)
        print(f"Wrote {args.write_default_rules}")
        return 0

    # Probe inputs before opening any of them.
    devices = list_devices()
    if args.list_devices:
        if not devices:
            print("No working video inputs found.")
            return 1
        for d in devices:
            print(f"{d['index']}: {d['name']}")
        return 0
    if not devices:
        log.error("No working video inputs found. Is the capture card plugged in?")
        return 1

    index = args.device
    if index is None:
        index = CAPTURE_INDEX if CAPTURE_INDEX in [d["index"] for d in devices] else devices[0]["index"]
    log.info("Using video input %d", index)

    screen = Screen(index, CAPTURE_SIZE, LOG_DIR)
    try:
        if args.snapshot:
            cv2.imwrite(args.snapshot, screen.frame())
            print(f"Saved {args.snapshot}")
            return 0
        if args.ocr_test:
            print(screen.text())
            return 0
        if args.check_rules:
            rules = load_rules(args.rules)
            if args.check_once:
                print(check_rules_once(screen, rules)[0])
                return 0
            # Keep the session alive: serve the live view (and web remote) so you can
            # click through screens while the rule matches update.
            try:
                chk_keys = KeySender(args.esp_url, dry_run=args.dry_run, token=args.esp_token)
                start_live_server(screen, devices, args.port, args.cert, args.key, chk_keys,
                                  args.token, RuleLearner(screen, chk_keys, args.rules or args.calibrate_out))
            except OSError as e:
                log.warning("Live view not started (%s); continuing without it", e)
            run_check_rules(screen, rules, args.check_interval)
            return 0
        if args.save_template:
            roi = None
            if args.roi:
                roi = tuple(int(v) for v in args.roi.split(","))
                if len(roi) != 4:
                    raise SystemExit("--roi must be X,Y,W,H")
            print(f"Saved {save_template(screen.frame(), args.save_template, roi)}")
            return 0

        keys = KeySender(args.esp_url, dry_run=args.dry_run, token=args.esp_token)

        if args.live or args.view_only or args.calibrate:
            # Learning mode is off during --calibrate, which rewrites its whole output file.
            learner = None if args.calibrate else RuleLearner(screen, keys, args.rules or args.calibrate_out)
            start_live_server(screen, devices, args.port, args.cert, args.key,
                              keys, args.token, learner)

        if args.calibrate:
            try:
                Calibrator(screen, keys, args.calibrate_out).run()
            except KeyboardInterrupt:
                log.info("Calibration stopped. Rules are in %s - review them, then run: "
                         "python xbox_updater.py --setup --rules %s",
                         args.calibrate_out, args.calibrate_out)
            return 0

        if args.view_only:
            set_status("view only")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0

        updater = Updater(keys, screen, visit_similarity=args.visit_similarity)

        if args.setup:
            ok = updater.run_setup(load_rules(args.rules), args.setup_max_steps)
            if ok and args.then_update:
                ok = updater.run_flow() and updater.check_and_update() in ("updated", "up_to_date")
            log.info("Setup %s", "finished" if ok else "did not finish")
            if args.live:
                log.info("Live view stays up. Ctrl+C to quit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            return 0 if ok else 1

        if not updater.run_flow():
            return 2
        result = updater.check_and_update()
        log.info("Result: %s", result)
        if args.live:
            log.info("Updater finished; live view stays up. Ctrl+C to quit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return 0 if result in ("updated", "up_to_date") else 1
    finally:
        screen.release()


if __name__ == "__main__":
    sys.exit(main())
