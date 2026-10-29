# Xbox controller WebUSB helper (xpad unbind)

Makes any Xbox-style controller that Linux's `xpad` driver supports usable by browser-based tools such as DriftGuard (WebUSB). Previously this only matched USB ID `045e:0b00`; it now works for any controller xpad claims (360, One, Series, Elite, and many third-party pads).

## What it does, exactly

Whenever you plug in a controller and the kernel's `xpad` driver binds to it:

1. **Trigger.** A udev rule fires on the driver "bind" event (`ACTION=="bind"`, `DRIVER=="xpad"`) and runs `/usr/local/bin/xbox-webusb-unbind.sh <interface>` through `systemd-run`, so it runs detached and doesn't block udev. The interface name looks like `1-1:1.0`.
2. **Pause check.** If `/etc/xbox-webusb-unbind.disabled` exists, the script logs that it's paused and exits, leaving the controller alone.
3. **Permissions.** It finds the controller's USB device node (`/dev/bus/usb/BBB/DDD`) and sets it to mode `666`, so your normal user (and Chromium) can open it without root.
4. **Wait.** It sleeps for `DELAY` seconds (default `2.5`). The controller needs a moment after connecting before its sticks work properly.
5. **Unbind.** It writes the interface name to `/sys/bus/usb/drivers/xpad/unbind`. This is the same as running `echo "1-1:1.0" | sudo tee /sys/bus/usb/drivers/xpad/unbind`, but the interface name is detected automatically. If the controller was unplugged during the wait, it does nothing.
6. **Log.** Every step is logged to `/var/log/xbox-webusb-unbind.log`, including the controller's USB ID.

With xpad unbound, the browser can claim the interface over WebUSB.

### Second rule: controllers xpad doesn't claim

Some controllers, such as an Xbox 360 controller on a Play & Charge cable (`045e:028f`), enumerate but are never claimed by xpad, so the bind event above never fires and the permissions would never be fixed. A second udev rule covers this: when any Microsoft (`045e`) vendor-specific USB interface is added, the script runs with `--perms-only`. It does steps 2 (pause check), 3 (permissions) and 6 (log) only. There is nothing to unbind, so it doesn't wait or touch xpad.

**Important side effect:** while this is installed, **every** xpad-supported controller you plug in is unbound from xpad about 2.5 seconds after connecting, so none of them will work as a normal gamepad in games or Steam. Use the pause switch below when you want to play, or uninstall.

## Files

| File | Installed to | Purpose |
|---|---|---|
| `90-xbox-webusb-unbind.rules` | `/etc/udev/rules.d/` | Triggers the script when xpad binds |
| `xbox-webusb-unbind.sh` | `/usr/local/bin/` | Fixes permissions, waits, unbinds xpad |
| `install.sh` | (not installed) | Installs the two files above |
| `uninstall.sh` | (not installed) | Removes everything |

## Install

From the folder containing these files:

```bash
sudo ./install.sh
```

Or manually:

```bash
sudo install -m 755 xbox-webusb-unbind.sh /usr/local/bin/
sudo install -m 644 90-xbox-webusb-unbind.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

Then **unplug and replug the controller**. Wait a few seconds, then use your browser tool.

If you have the older single-device version installed (`50-usb-045e-0b00.rules` / `on-usb-045e-0b00.sh`), `install.sh` removes it automatically. Manually: `sudo rm -f /etc/udev/rules.d/50-usb-045e-0b00.rules /etc/udev/rules.d/99-usb-045e-0b00.rules /usr/local/bin/on-usb-045e-0b00.sh`.

## Check that it worked

```bash
lsusb -t                              # controller interface should NOT show Driver=xpad
tail /var/log/xbox-webusb-unbind.log  # should show "unbound ... from xpad"
```

## Pause and resume (play games normally without uninstalling)

```bash
sudo touch /etc/xbox-webusb-unbind.disabled   # pause: controllers stay bound to xpad
sudo rm /etc/xbox-webusb-unbind.disabled      # resume
```

Replug the controller after changing this; it's checked each time one connects.

## Adjusting the delay

Edit `DELAY=2.5` near the top of `/usr/local/bin/xbox-webusb-unbind.sh`:

```bash
sudo nano /usr/local/bin/xbox-webusb-unbind.sh
```

Earlier testing on an Elite Series 2: `5` worked, no delay left the sticks unresponsive, `2.5` worked well. Other controllers may differ. Changes take effect on the next plug-in, no reload needed.

## Uninstall (restore normal behaviour)

```bash
sudo ./uninstall.sh
```

Or manually:

```bash
sudo rm -f /etc/udev/rules.d/90-xbox-webusb-unbind.rules
sudo rm -f /usr/local/bin/xbox-webusb-unbind.sh
sudo rm -f /var/log/xbox-webusb-unbind.log
sudo rm -f /etc/xbox-webusb-unbind.disabled
sudo udevadm control --reload-rules
```

Then **unplug and replug the controller**. xpad will bind again and it will work as a normal gamepad.

To re-bind xpad without replugging (replace `1-1:1.0` with your interface from `lsusb -t`):

```bash
echo "1-1:1.0" | sudo tee /sys/bus/usb/drivers/xpad/bind
```

## Troubleshooting

- **Nothing in the log:** the controller's USB interface may not be vendor-specific class from Microsoft (`045e`) and may not be claimed by `xpad` (for example, if the `xone` driver claims it instead). Check `lsusb -t` to see which driver owns it. You can also watch events live with `udevadm monitor --environment --udev`, then replug. For a non-Microsoft controller that xpad doesn't claim, you can still fix permissions by hand: `sudo chmod 666 /dev/bus/usb/BBB/DDD`.
- **Browser can't open the device:** run `lsusb` to get the bus and device numbers, then `ls -l /dev/bus/usb/BBB/DDD` and check the permissions are `crw-rw-rw-`. Replug after installing.
- **Sticks don't register:** increase `DELAY`.
- **Chromium installed as Flatpak/Snap:** the sandbox can block raw USB access. Flatpak needs `--device=all`; a `.deb` Chromium/Chrome avoids this.
- **Other programs holding the controller** (Steam, `xboxdrv`, `xone`, a second browser tab): close them fully.
- **Wireless controllers (Bluetooth):** these don't use USB or xpad, so this does nothing for them. Connect by USB cable.
