/*
 * WiFi-controlled USB HID keyboard for LOLIN S3 (ESP32-S3).
 * Plug the board's native USB port into the Xbox One; send key taps over HTTP.
 *
 * API (matches xbox_updater.py):
 *   POST /key   body: {"key":"down"}        -> taps one key
 *   GET  /key?key=down                      -> same, handy for testing in a browser
 *   GET  /keys                              -> lists supported key names
 *   GET  /                                  -> status
 *
 * Arduino IDE settings (Tools menu):
 *   Board:              LOLIN S3   (or "ESP32S3 Dev Module")
 *   USB Mode:           USB-OTG (TinyUSB)
 *   USB CDC On Boot:    Disabled
 *   Upload Mode:        UART0 / Hardware CDC (default)
 *
 * Needs the "esp32" board package by Espressif (2.x or 3.x). No extra libraries.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include "USB.h"
#include "USBHIDKeyboard.h"

// ---- Config ----------------------------------------------------------------
const char* WIFI_SSID  = "GL-SFT1200-487";
const char* WIFI_PASS  = "goodlife";
const char* HOSTNAME   = "xbox-keyboard";   // reachable as http://xbox-keyboard.local
const char* AUTH_TOKEN = "";                // optional; if set, send header "X-Token: <value>"
const uint16_t KEY_HOLD_MS = 40;            // how long each key is held down
// ----------------------------------------------------------------------------

USBHIDKeyboard Keyboard;
WebServer server(80);

struct KeyMap {
  const char* name;
  uint8_t code;
};

// "guide" sends the Windows/GUI key. Whether the Xbox treats it like the Guide
// button depends on the dashboard version - verify on your console.
const KeyMap KEYS[] = {
  {"up",        KEY_UP_ARROW},
  {"down",      KEY_DOWN_ARROW},
  {"left",      KEY_LEFT_ARROW},
  {"right",     KEY_RIGHT_ARROW},
  {"enter",     KEY_RETURN},
  {"esc",       KEY_ESC},
  {"tab",       KEY_TAB},
  {"space",     ' '},
  {"backspace", KEY_BACKSPACE},
  {"guide",     KEY_LEFT_GUI},
};
const size_t KEY_COUNT = sizeof(KEYS) / sizeof(KEYS[0]);

bool lookupKey(const String& name, uint8_t& code) {
  String n = name;
  n.toLowerCase();
  n.trim();
  for (size_t i = 0; i < KEY_COUNT; i++) {
    if (n == KEYS[i].name) {
      code = KEYS[i].code;
      return true;
    }
  }
  return false;
}

// Pulls the string value for "key" out of a tiny JSON body, no JSON library needed.
String extractJsonKey(const String& body) {
  int k = body.indexOf("\"key\"");
  if (k < 0) return "";
  int colon = body.indexOf(':', k);
  if (colon < 0) return "";
  int q1 = body.indexOf('"', colon + 1);
  if (q1 < 0) return "";
  int q2 = body.indexOf('"', q1 + 1);
  if (q2 < 0) return "";
  return body.substring(q1 + 1, q2);
}

bool authorized() {
  if (strlen(AUTH_TOKEN) == 0) return true;
  return server.header("X-Token") == AUTH_TOKEN;
}

void handleKey() {
  if (!authorized()) {
    server.send(401, "text/plain", "unauthorized");
    return;
  }

  String key = server.arg("key");                 // query string or form field
  if (key.isEmpty() && server.hasArg("plain")) {  // raw JSON body
    key = extractJsonKey(server.arg("plain"));
  }
  if (key.isEmpty()) {
    server.send(400, "text/plain", "missing key");
    return;
  }

  uint8_t code;
  if (!lookupKey(key, code)) {
    server.send(400, "text/plain", "unknown key: " + key);
    return;
  }

  Keyboard.press(code);
  delay(KEY_HOLD_MS);
  Keyboard.releaseAll();
  server.send(200, "application/json", "{\"ok\":true,\"key\":\"" + key + "\"}");
}

void handleKeys() {
  String out = "[";
  for (size_t i = 0; i < KEY_COUNT; i++) {
    if (i) out += ",";
    out += "\"" + String(KEYS[i].name) + "\"";
  }
  out += "]";
  server.send(200, "application/json", out);
}

void handleRoot() {
  String s = "xbox-keyboard online\nip: " + WiFi.localIP().toString() +
             "\nrssi: " + String(WiFi.RSSI()) + " dBm\nuptime: " +
             String(millis() / 1000) + " s\n";
  server.send(200, "text/plain", s);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.setSleep(false);               // lower latency for key taps
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
  }
}

void setup() {
  Keyboard.begin();
  USB.begin();

  connectWiFi();

  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
  }

  const char* headers[] = {"X-Token"};
  server.collectHeaders(headers, 1);
  server.on("/", HTTP_GET, handleRoot);
  server.on("/keys", HTTP_GET, handleKeys);
  server.on("/key", HTTP_GET, handleKey);
  server.on("/key", HTTP_POST, handleKey);
  server.begin();
}

void loop() {
  // Reconnect if WiFi drops (router reboot, etc.)
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    MDNS.begin(HOSTNAME);
  }
  server.handleClient();
  delay(1);
}
