"""Virtual gamepad interface.

Wraps the vgamepad library to provide a clean button/stick API.
Also provides a NullGamepad for testing without hardware.

IMPORTANT: This is for OFFLINE SANDBOX TESTING ONLY.

FIX NOTES:
- Uses integer API (left_joystick / right_joystick) with explicit
  scaling to [-32768, 32767] instead of the float API which silently
  produces zero on some Windows builds / vgamepad versions.
- Tracks all state so reset() actually releases everything.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# Button name → vgamepad constant mapping (Xbox 360 layout)
_BUTTON_MAP: dict[str, int] = {}

_VGAMEPAD_AVAILABLE = False
try:
    import vgamepad as vg  # type: ignore[import-untyped]

    _VGAMEPAD_AVAILABLE = True
    _BUTTON_MAP = {
        "A": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "X": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "LB": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        "RB": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        "LT_DIGITAL": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        "RT_DIGITAL": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
        "START": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
        "BACK": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        "DPAD_UP": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        "DPAD_DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        "DPAD_LEFT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        "DPAD_RIGHT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        "LS_CLICK": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        "RS_CLICK": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
    }
except ImportError:
    pass


def _float_to_short(value: float) -> int:
    """Convert [-1.0, 1.0] float to Xbox stick integer [-32768, 32767].

    This is the critical scaling step — vgamepad's left_joystick()
    expects raw signed 16-bit integers.
    """
    clamped = max(-1.0, min(1.0, value))
    if clamped >= 0:
        return int(clamped * 32767)
    return int(clamped * 32768)


def _float_to_trigger(value: float) -> int:
    """Convert [0.0, 1.0] float to Xbox trigger byte [0, 255]."""
    return int(max(0.0, min(1.0, value)) * 255)


class GamepadInterface:
    """Virtual Xbox 360 gamepad for offline sandbox testing.

    Translates named buttons and normalized stick values to vgamepad calls.
    Falls back to NullGamepad if vgamepad is not installed.
    """

    def __init__(self) -> None:
        if not _VGAMEPAD_AVAILABLE:
            logger.warning("vgamepad not available — using NullGamepad")
            self._pad: _GamepadBackend = _NullPad()
        else:
            self._pad = _VGamepadPad()

        self._pressed: set[str] = set()
        self._left_stick: tuple[float, float] = (0.0, 0.0)
        self._right_stick: tuple[float, float] = (0.0, 0.0)

    def press_button(self, button: str) -> None:
        if button in self._pressed:
            logger.debug("gamepad.press ignored_already_pressed button=%s", button)
            return
        self._pressed.add(button)
        logger.debug("gamepad.press button=%s pressed=%s", button, sorted(self._pressed))
        self._pad.press(button)

    def release_button(self, button: str) -> None:
        self._pressed.discard(button)
        logger.debug("gamepad.release button=%s pressed=%s", button, sorted(self._pressed))
        self._pad.release(button)

    def set_left_stick(self, x: float, y: float) -> None:
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._left_stick = (x, y)
        logger.debug("gamepad.left_stick x=%.4f y=%.4f", x, y)
        self._pad.left_stick(x, y)

    def set_right_stick(self, x: float, y: float) -> None:
        """Right stick / ``right_joystick_x`` / ``right_joystick_y`` (normalized → int16).

        Used for Pro Switching (flick toward a teammate) and any manual RS aiming.
        """
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._right_stick = (x, y)
        logger.debug("gamepad.right_stick x=%.4f y=%.4f", x, y)
        self._pad.right_stick(x, y)

    def reset(self) -> None:
        """Release all buttons and center sticks."""
        logger.info("gamepad.reset pressed=%s", sorted(self._pressed))
        for button in list(self._pressed):
            self.release_button(button)
        self.set_left_stick(0.0, 0.0)
        self.set_right_stick(0.0, 0.0)
        self._pad.reset_triggers()

    def update(self) -> None:
        """Commit current state to the virtual device."""
        logger.debug(
            "gamepad.update left=%s right=%s pressed=%s",
            self._left_stick,
            self._right_stick,
            sorted(self._pressed),
        )
        self._pad.commit()


class _GamepadBackend(Protocol):
    def press(self, button: str) -> None: ...

    def release(self, button: str) -> None: ...

    def left_stick(self, x: float, y: float) -> None: ...

    def right_stick(self, x: float, y: float) -> None: ...

    def reset_triggers(self) -> None: ...

    def commit(self) -> None: ...


class _VGamepadPad:
    """Real vgamepad backend — uses integer API for reliable stick input."""

    def __init__(self) -> None:
        self._gp = vg.VX360Gamepad()  # type: ignore[name-defined]
        logger.info("Virtual Xbox 360 gamepad created")

    def press(self, button: str) -> None:
        if button == "LT":
            self._gp.left_trigger(value=_float_to_trigger(1.0))
        elif button == "RT":
            self._gp.right_trigger(value=_float_to_trigger(1.0))
        elif button == "RS_CLICK":
            self._gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)  # type: ignore[name-defined]
        else:
            mapped = _BUTTON_MAP.get(button)
            if mapped is not None:
                self._gp.press_button(button=mapped)
            else:
                logger.warning(f"Unknown button: {button}")

    def release(self, button: str) -> None:
        if button == "LT":
            self._gp.left_trigger(value=0)
        elif button == "RT":
            self._gp.right_trigger(value=0)
        elif button == "RS_CLICK":
            self._gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)  # type: ignore[name-defined]
        else:
            mapped = _BUTTON_MAP.get(button)
            if mapped is not None:
                self._gp.release_button(button=mapped)

    def left_stick(self, x: float, y: float) -> None:
        # Use INTEGER API — the float API silently produces 0 on some builds
        ix = _float_to_short(x)
        iy = _float_to_short(y)
        self._gp.left_joystick(x_value=ix, y_value=iy)

    def right_stick(self, x: float, y: float) -> None:
        """XInput right thumbstick (Pro Switch / skill moves)."""
        ix = _float_to_short(x)
        iy = _float_to_short(y)
        self._gp.right_joystick(x_value=ix, y_value=iy)

    def reset_triggers(self) -> None:
        self._gp.left_trigger(value=0)
        self._gp.right_trigger(value=0)

    def commit(self) -> None:
        self._gp.update()


class _NullPad:
    """No-op gamepad for testing or when vgamepad is unavailable."""

    def press(self, button: str) -> None:
        pass

    def release(self, button: str) -> None:
        pass

    def left_stick(self, x: float, y: float) -> None:
        pass

    def right_stick(self, x: float, y: float) -> None:
        pass

    def reset_triggers(self) -> None:
        pass

    def commit(self) -> None:
        pass
