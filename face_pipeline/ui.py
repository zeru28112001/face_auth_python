import cv2
import numpy as np


def draw_alert_card(
    frame: np.ndarray,
    title: str,
    message: str,
    hint: str = "",
    alert_type: str = "ERROR",  # "ERROR", "SUCCESS", "INFO"
) -> None:
    """Draws a sleek, modern, minimal floating notification card near the top of the feed."""
    h, w = frame.shape[:2]

    card_w = min(int(w * 0.86), 540)
    card_h = 74 if not hint else 96
    x1 = (w - card_w) // 2
    y1 = 25
    x2 = x1 + card_w
    y2 = y1 + card_h

    if alert_type == "ERROR":
        accent_color = (60, 60, 235)    # Modern Red
    elif alert_type == "SUCCESS":
        accent_color = (80, 205, 60)    # Mint Green
    else:
        accent_color = (30, 165, 245)   # Amber / Blue

    # 1. Dark translucent card background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (22, 22, 26), cv2.FILLED)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # 2. Sleek border & left accent stripe
    cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 55, 65), 1)
    cv2.rectangle(frame, (x1, y1), (x1 + 5, y2), accent_color, cv2.FILLED)

    # 3. Title badge
    cv2.putText(
        frame,
        f"● {title.upper()}",
        (x1 + 18, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        accent_color,
        2,
        cv2.LINE_AA,
    )

    # 4. Message line
    cv2.putText(
        frame,
        message,
        (x1 + 18, y1 + 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    # 5. Hint subtext
    if hint:
        cv2.putText(
            frame,
            hint,
            (x1 + 18, y1 + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 170),
            1,
            cv2.LINE_AA,
        )
