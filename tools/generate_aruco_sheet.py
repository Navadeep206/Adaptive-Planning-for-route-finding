#!/usr/bin/env python3
"""Generate printable/displayable ArUco fiducial calibration markers.

Generates 4 high-contrast ArUco markers (DICT_4X4_50):
- ID 0: Top-Left (Far-Left)     -> (0.0, 0.0) cm
- ID 1: Top-Right (Far-Right)   -> (100.0, 0.0) cm
- ID 2: Bottom-Right (Near-Right)-> (100.0, 80.0) cm
- ID 3: Bottom-Left (Near-Left)  -> (0.0, 80.0) cm

Outputs:
- assets/aruco_tabletop_sheet.png (Full 4-corner calibration layout)
- assets/markers/marker_id_0_TL.png
- assets/markers/marker_id_1_TR.png
- assets/markers/marker_id_2_BR.png
- assets/markers/marker_id_3_BL.png
"""

import os
import cv2
import numpy as np


def generate_aruco_assets(output_dir: str = "assets") -> None:
    """Generate ArUco calibration sheet and individual marker images."""
    markers_dir = os.path.join(output_dir, "markers")
    os.makedirs(markers_dir, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 200
    border_bits = 1

    marker_meta = [
        (0, "TL (Top-Left / Far-Left)", "marker_id_0_TL.png", (0.0, 0.0)),
        (1, "TR (Top-Right / Far-Right)", "marker_id_1_TR.png", (100.0, 0.0)),
        (2, "BR (Bottom-Right / Near-Right)", "marker_id_2_BR.png", (100.0, 80.0)),
        (3, "BL (Bottom-Left / Near-Left)", "marker_id_3_BL.png", (0.0, 80.0)),
    ]

    # Generate individual markers with white padding for quiet zone
    marker_imgs = {}
    for marker_id, label, filename, coord in marker_meta:
        # Raw marker
        raw_marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, borderBits=border_bits)
        # Pad with white quiet zone (40px)
        pad = 40
        padded = np.full((marker_size + 2 * pad, marker_size + 2 * pad), 255, dtype=np.uint8)
        padded[pad:pad + marker_size, pad:pad + marker_size] = raw_marker

        # Add text label below marker
        padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        label_text = f"ID {marker_id}: {label}"
        coord_text = f"Coord: ({coord[0]:.0f}, {coord[1]:.0f}) cm"
        cv2.putText(padded_bgr, label_text, (20, padded_bgr.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(padded_bgr, coord_text, (20, padded_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1, cv2.LINE_AA)

        out_path = os.path.join(markers_dir, filename)
        cv2.imwrite(out_path, padded_bgr)
        marker_imgs[marker_id] = padded_bgr
        print(f"[OK] Saved individual marker: {out_path}")

    # Generate 1920x1080 tabletop reference layout sheet
    canvas_w, canvas_h = 1920, 1080
    sheet = np.full((canvas_h, canvas_w, 3), 250, dtype=np.uint8)

    # Title header
    cv2.putText(sheet, "ARUCO DYNAMIC CALIBRATION SHEET (DICT_4X4_50)", (canvas_w // 2 - 380, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(sheet, "Place on table or open on display. System auto-calibrates in real-time.",
                (canvas_w // 2 - 330, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

    # Grid guide lines connecting the 4 corners
    pad_x, pad_y = 100, 120
    m_h, m_w = marker_imgs[0].shape[:2]

    # Draw plane perimeter box
    cv2.rectangle(sheet, (pad_x + m_w // 2, pad_y + m_h // 2),
                  (canvas_w - pad_x - m_w // 2, canvas_h - pad_y - m_h // 2),
                  (200, 200, 200), 2, cv2.LINE_AA)

    # Place Top-Left (ID 0)
    sheet[pad_y:pad_y + m_h, pad_x:pad_x + m_w] = marker_imgs[0]

    # Place Top-Right (ID 1)
    sheet[pad_y:pad_y + m_h, canvas_w - pad_x - m_w:canvas_w - pad_x] = marker_imgs[1]

    # Place Bottom-Right (ID 2)
    sheet[canvas_h - pad_y - m_h:canvas_h - pad_y, canvas_w - pad_x - m_w:canvas_w - pad_x] = marker_imgs[2]

    # Place Bottom-Left (ID 3)
    sheet[canvas_h - pad_y - m_h:canvas_h - pad_y, pad_x:pad_x + m_w] = marker_imgs[3]

    # Center target icon
    cx, cy = canvas_w // 2, canvas_h // 2
    cv2.circle(sheet, (cx, cy), 12, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.drawMarker(sheet, (cx, cy), (180, 180, 180), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(sheet, "SURFACE CENTER (50.0, 40.0) cm", (cx - 130, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1, cv2.LINE_AA)

    sheet_path = os.path.join(output_dir, "aruco_tabletop_sheet.png")
    cv2.imwrite(sheet_path, sheet)
    print(f"[OK] Saved full calibration sheet: {sheet_path}")


if __name__ == "__main__":
    generate_aruco_assets()

