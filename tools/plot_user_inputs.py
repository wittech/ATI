# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image, ImageDraw
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import yaml, argparse, os
import math


def plot_tracks(
    img: Image.Image,
    tracks: np.ndarray,
    line_width: int = 10,
    dot_radius: int = 10,
    arrow_length: int = 25,
    arrow_angle_deg: float = 30.0
) -> Image.Image:
    """
    Plot trajectories on an image, with a dot at the start and an arrow whose center
    aligns with the last visible trajectory point.

    Args:
        img: A PIL Image.
        tracks: Array of shape (N, T, 1, 3): (x, y, visibility).
        line_width: Thickness of trajectory lines.
        dot_radius: Radius of the start dot.
        arrow_length: Length of each arrowhead side.
        arrow_angle_deg: Angle between shaft and arrowhead sides (degrees).
    """
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    N, T, _, _ = tracks.shape
    arrow_angle = math.radians(arrow_angle_deg)

    for i in range(N):
        traj = tracks[i, :, 0, :]
        if traj.shape[-1] == 4:
            traj = np.concatenate([traj[..., :2], traj[..., -1:]], axis=-1)
        # Draw segments
        for t in range(T - 1):
            x1, y1, v1 = traj[t]
            x2, y2, v2 = traj[t + 1]
            if v1 == 0 or v2 == 0:
                continue
            ratio = t / (T - 1)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 30)
            draw.line([(int(x1), int(y1)), (int(x2), int(y2))],
                      fill=color, width=line_width)

        # Visible indices
        visible = [t for t in range(T) if traj[t, 2] == 1]
        if not visible:
            continue

        # Start dot
        t0 = visible[0]
        x0, y0, _ = traj[t0]
        draw.ellipse([
            (int(x0 - dot_radius), int(y0 - dot_radius)),
            (int(x0 + dot_radius), int(y0 + dot_radius))
        ], fill=(0, 255, 30))

        # Arrow at end
        t_last = visible[-1]
        ratio_last = t_last / (T - 1)
        arrow_color = (int(255 * ratio_last), int(255 * (1 - ratio_last)), 30)

        # Direction: average of last two segments if available
        if len(visible) >= 3:
            t2, t1, tL = visible[-3], visible[-2], visible[-1]
            x2, y2, _ = traj[t2]
            x1, y1, _ = traj[t1]
            xL, yL, _ = traj[tL]
            v1 = (x1 - x2, y1 - y2)
            v2 = (xL - x1, yL - y1)
            dx, dy = (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2
        else:
            x1, y1, _ = traj[visible[-2]]
            xL, yL, _ = traj[t_last]
            dx, dy = xL - x1, yL - y1

        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            continue
        ux, uy = dx / dist, dy / dist

        # Arrowhead points
        def rotate(vx, vy, ang):
            return vx * math.cos(ang) - vy * math.sin(ang), vx * math.sin(ang) + vy * math.cos(ang)

        vx1, vy1 = rotate(ux, uy,  arrow_angle)
        vx2, vy2 = rotate(ux, uy, -arrow_angle)
        p1 = (xL - vx1 * arrow_length, yL - vy1 * arrow_length)
        p2 = (xL - vx2 * arrow_length, yL - vy2 * arrow_length)

        # Compute translation to center triangle on (xL, yL)
        cx = (xL + p1[0] + p2[0]) / 3
        cy = (yL + p1[1] + p2[1]) / 3
        dx_c, dy_c = xL - cx, yL - cy

        tip = (xL + dx_c, yL + dy_c)
        p1_c = (p1[0] + dx_c, p1[1] + dy_c)
        p2_c = (p2[0] + dx_c, p2[1] + dy_c)

        draw.polygon([tip, p1_c, p2_c], fill=arrow_color)

    return canvas


def unzip_to_array(
    data: bytes, key: Union[str, List[str]] = "array"
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    bytes_io = io.BytesIO(data)

    if isinstance(key, str):
        # Load the NPZ data from the BytesIO object
        with np.load(bytes_io) as data:
            return data[key]
    else:
        get = {}
        with np.load(bytes_io) as data:
            for k in key:
                get[k] = data[k]
        return get


def main():
    parser = argparse.ArgumentParser(description="Plot trajectories on images")
    parser.add_argument("base_file", help="Path to YAML file describing images and tracks")
    parser.add_argument("--save_dir", default='', type=str, help="Path save images")
    args = parser.parse_args()

    # Load YAML list of dicts
    with open(args.base_file, 'r') as f:
        items = yaml.safe_load(f)  # List[Dict]

    for ii, item in enumerate(items):
        image_path = item["image"]
        track_path = item["track"]

        # Load image and tracks
        img = Image.open(image_path)
        raw_tracks = torch.load(track_path)
        tracks = unzip_to_array(raw_tracks) / 8

        # import ipdb; ipdb.set_trace()

        # Plot trajectories
        try:
            out_img = plot_tracks(img, tracks,)
        except Exception as e:
            print(f"Error plotting tracks for {image_path}: {e}")
            continue
        
        if not args.save_dir:
            # Determine output path
            out_path = image_path.replace("/images/", "/images_track_input/")
        else:
            out_path = os.path.join(args.save_dir, f'{ii:02d}.jpg')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Save output image
        out_img.save(out_path)
        print(f"Saved plotted image to {out_path}")


if __name__ == "__main__":
    main()
