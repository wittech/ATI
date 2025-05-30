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

import cv2
import mediapy as media
import torch
import os
import tqdm
import argparse
import numpy as np
import yaml
import random
import colorsys
from typing import Dict, List, Tuple, Optional
import io
from typing import Union


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


# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors


def age_to_bgr(ratio: float) -> Tuple[int,int,int]:
    """
    Map ratio∈[0,1] through: 0→blue, 1/3→green, 2/3→yellow, 1→red.
    Returns (B,G,R) for OpenCV.
    """
    if ratio <= 1/3:
        # blue→green
        t = ratio / (1/3)
        b = int(255 * (1 - t))
        g = int(255 * t)
        r = 0
    elif ratio <= 2/3:
        # green→yellow
        t = (ratio - 1/3) / (1/3)
        b = 0
        g = 255
        r = int(255 * t)
    else:
        # yellow→red
        t = (ratio - 2/3) / (1/3)
        b = 0
        g = int(255 * (1 - t))
        r = 255
    return (r, g, b)


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    min_radius: int = 1,
    max_radius: int = 6,
    max_retain: int = 50
) -> np.ndarray:
    """
    Draws every past point of each track on each frame, with radius and color
    interpolated by the point's age (old→small to new→large).

    Args:
      frames:      [F, H, W, 3] uint8 RGB
      point_tracks:[N, F, 2] float32  – (x,y) in pixel coords
      visibles:    [N, F] bool        – visibility mask
      min_radius:  radius for the very first point (oldest)
      max_radius:  radius for the current point (newest)

    Returns:
      video: [F, H, W, 3] uint8 RGB
    """
    num_points, num_frames = point_tracks.shape[:2]
    H, W = frames.shape[1:3]

    video = frames.copy()

    for t in range(num_frames):
        # start from the original frame
        frame = video[t].copy()

        for i in range(num_points):
            # draw every past step τ = 0..t
            for τ in range(t + 1):
                if not visibles[i, τ]:
                    continue

                if t - τ > max_retain:
                    continue

                # sub-pixel offset + clamp
                x, y = point_tracks[i, τ] + 0.5
                xi = int(np.clip(x, 0, W - 1))
                yi = int(np.clip(y, 0, H - 1))

                # age‐ratio in [0,1]
                if num_frames > 1:
                    ratio = 1 - float(t - τ) / max_retain
                else:
                    ratio = 1.0

                # interpolated radius
                radius = int(round(min_radius + (max_radius - min_radius) * ratio))

                # OpenCV draws in BGR order:
                color_rgb = age_to_bgr(ratio)

                # filled circle
                cv2.circle(frame, (xi, yi), radius, color_rgb, thickness=-1)

        video[t] = frame

    return video


parser = argparse.ArgumentParser(
    description="Visualize tracks."
)
parser.add_argument(
    "--base_dir",
    type=str,
    default='samples',
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="outputs",
)
parser.add_argument(
    "--track_dir",
    type=str,
    default="tracks",
)
parser.add_argument(
    "--output_appendix",
    type=str,
    default="_vis",
)

args = parser.parse_args()

base_dir = args.base_dir
video_dir = os.path.join(base_dir, args.video_dir)
track_dir = os.path.join(base_dir, args.track_dir)
os.makedirs(video_dir + args.output_appendix, exist_ok=True)

print([t for t in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, t))])
while len([t for t in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, t))]) == 1:
    video_dir = os.path.join(video_dir, [t for t in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, t))][0])
print("Source:", video_dir)

shift_ = 3

records = yaml.safe_load(open(os.path.join(base_dir, 'test.yaml'), 'r'))

for video_name in tqdm.tqdm(os.listdir(video_dir)):
    if '.mp4' not in video_name:
        continue
    nn = os.path.basename(video_name)
    nn = int(nn.split('.')[0] if '_' not in nn else nn.split('_')[0])

    video = media.read_video(os.path.join(video_dir, video_name))

    short_edge = min(*video.shape[1:3])
    H, W = video.shape[1:3]

    track = torch.load(records[nn]['track'])
    if isinstance(track, bytes):
        track = unzip_to_array(track)
        track = np.repeat(track, 2, axis=1)[:, ::3]
        points = track[:, :, 0, :2].astype(np.float32) / 8
        visibles = track[:, :, 0, 2].astype(np.float32) / 8

        # image_origin = os.path.join(base_dir, 'images', f'{nn:02d}.png')
        image_origin = records[nn]['image']
        image = media.read_image(image_origin)

        H_ori, W_ori, _ = image.shape
        points = points / np.array([W_ori, H_ori]) * np.array([W, H])

    else:
        points = (track[shift_:, :, :2] + track[shift_:, :, 2:4]) / 2 * short_edge + torch.tensor([W / 2, H / 2])
        visibles = track[shift_:, :, -1]

        points = torch.permute(points, (1, 0, 2)).cpu().numpy()
        visibles = torch.permute(visibles, (1, 0)).cpu().numpy()

    video_viz = paint_point_track(video, points, visibles)
    name_ = os.path.basename(video_name).split('.')[0]
    media.write_video(os.path.join(base_dir, args.video_dir + args.output_appendix, f'{name_}_viz.mp4'), video_viz, fps=16)