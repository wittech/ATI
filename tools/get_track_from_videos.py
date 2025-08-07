import torch
from typing import List, Sequence, Any
from PIL import Image
import numpy as np
import cv2
import yaml
import math
import io


TAP_MODELS = {
    'bootstapir': 'projects/video_any/debug/checkpoints/bootstapir_checkpoint_v2.pt',
}

QUANT_MULTI = 8
def array_to_npz_bytes(arr, path, compressed=True, quant_multi=QUANT_MULTI):
    # pack into uint16 as before
    arr_q = (quant_multi * arr).astype(np.float32)
    bio = io.BytesIO()
    if compressed:
        np.savez_compressed(bio, array=arr_q)
    else:
        np.savez(bio, array=arr_q)
    torch.save(bio.getvalue(), path)


def parse_to_list(text: str) -> List[List[int]]:
    """
    Parse a multiline string of comma-separated integers into a list of integer lists.

    Example:
        text = "327, 806, 670, 1164\n49, 587, 346, 1037"
        parse_to_list(text)
        # → [[327, 806, 670, 1164], [49, 587, 346, 1037]]
    """
    lines = text.strip().splitlines()
    result: List[List[int]] = []
    for line in lines:
        # split on comma, strip whitespace, convert to int
        nums = [int(x.strip()) for x in line.split(',') if x.strip()]
        if nums:
            result.append(nums)
    return result



def load_video_to_frames(
    video_path: str,
    preset_fps: float = 24,
    max_short_edge: int = None
) -> List[Image.Image]:
    """
    Load a video file, resample its frame-rate to a single preset value
    (if needed), optionally resize frames so their short edge is at most
    max_short_edge (keeping aspect ratio), and return a list of PIL.Image frames.

    Args:
        video_path (str): Path to the video file.
        preset_fps (float): Desired FPS. If the video's FPS isn't exactly
            this value, the video will be resampled to match it.
        max_short_edge (int, optional): If provided and a frame's short edge
            (min(width,height)) exceeds this, the frame is resized so the
            short edge == max_short_edge, preserving aspect ratio.

    Returns:
        List[PIL.Image.Image]: A list of frames at the preset FPS, each
            resized if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    do_resample = fps_in > 0 and abs(fps_in - preset_fps) > 1e-3

    # read all frames
    raw_frames: List[Image.Image] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # optional resize by short edge
        if max_short_edge is not None:
            w, h = img.size
            short_edge = min(w, h)
            if short_edge > max_short_edge:
                scale = max_short_edge / short_edge
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)

        raw_frames.append(img)
    cap.release()

    # resample FPS if needed
    if do_resample:
        ratio = fps_in / preset_fps
        total_in = len(raw_frames)
        total_out = int(math.floor(total_in / ratio))
        resampled: List[Image.Image] = []
        for i in range(total_out):
            idx = min(int(round(i * ratio)), total_in - 1)
            resampled.append(raw_frames[idx])
        return resampled

    return raw_frames


def sample_grid_points(bbox, N):
    """
    Uniformly sample N points inside a bounding box using a grid
    whose Nx×Ny layout follows the box’s width:height ratio.

    Args:
        bbox: tuple (ymin, xmin, ymax, xmax)
        N:     int, number of points to sample

    Returns:
        numpy.ndarray of shape (N, 2), each row is (y, x)
    """
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    # choose Nx and Ny so that Nx/Ny ≈ width/height and Nx*Ny >= N
    Nx = int(np.ceil(np.sqrt(N * width / height)))
    Ny = int(np.ceil(np.sqrt(N * height / width)))

    # generate evenly spaced coordinates along each axis
    ys = np.linspace(ymin, ymax, Ny)
    xs = np.linspace(xmin, xmax, Nx)

    # form the grid and flatten
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)

    # return exactly N samples
    return coords



def resize_images_to_size(image_list, size=1024):
    """
    Given a list of PIL Image objects, resize each so that
    width and height are multiples of 16, using nearest multiple rounding.
    Returns a new list of resized images.
    """
    resized_list = []
    for img in image_list:        
        # Resize using a high-quality resample filter (e.g. LANCZOS).
        # You can also use Image.BILINEAR, Image.BICUBIC, etc.
        resized_img = img.resize((size, size), resample=Image.LANCZOS)
        resized_list.append(resized_img)
    
    return resized_list


def resize_box(box, ratios):
    return [int(round(box[0] * ratios[0])), int(round(box[1] * ratios[1])), int(round(box[2] * ratios[0])), int(round(box[3] * ratios[1]))]


class TrackAnyPoint():
    def __init__(self, n_points=60):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_points = n_points
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
        self.resolution = 720
        self.boundary_remove = 40

    @torch.no_grad()
    def __call__(self, video_frames: List[Image.Image]):
        ori_w, ori_h = video_frames[0].size
        video_frames = resize_images_to_size(video_frames, size=self.resolution)

        boxes = [[self.boundary_remove, self.boundary_remove, video_frames[0].size[0] - self.boundary_remove, video_frames[0].size[1] - self.boundary_remove]]

        representative_points = [torch.from_numpy(sample_grid_points(box, int(self.n_points / len(boxes)))).to(self.device) for box in boxes]
        representative_points = torch.cat(representative_points, dim=0)
        representative_points = torch.cat([torch.zeros_like(representative_points[..., :1]), representative_points], dim=-1)
        frames_np = [np.array(frame) for frame in video_frames]

        get_trackers = self.inference(np.array(frames_np), ori_w, ori_h, representative_points[None])

        return get_trackers

    @torch.no_grad()
    def inference(self, frames: np.ndarray, w_ori, h_ori, tracks) -> np.ndarray:
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
        _, _, _, H, W = video.shape

        tracks = tracks.float()

        # Run inference. The forward now returns a mapping, e.g., with key 'pred_tracks'.
        tracks, visibles = self.model(video, tracks)
        # Extract predicted tracks; expected shape is (B, T, N, 2)
        tracks = convert_grid_coordinates(tracks, (W, H), (w_ori, h_ori),)

        # [H, W, T, 2] => [T, H, W, 2]
        return torch.cat([tracks, visibles.unsqueeze(-1)], dim=-1).cpu().numpy()


def convert_grid_coordinates(
    coords: torch.Tensor,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> torch.Tensor:
    """
    Convert image coordinates between image grids of different sizes using PyTorch.

    By default, the function assumes that the image corners are aligned.
    It scales the coordinates from the input grid to the output grid by multiplying
    by the size ratio.

    Args:
        coords (torch.Tensor): The coordinates to be converted.
            For 'xy', the tensor should have shape [..., 2].
            For 'tyx', the tensor should have shape [..., 3].
        input_grid_size (Sequence[int]): The size of the current grid.
            For 'xy', it should be [width, height].
            For 'tyx', it should be [num_frames, height, width].
        output_grid_size (Sequence[int]): The size of the target grid.
            For 'xy', it should be [width, height].
            For 'tyx', it should be [num_frames, height, width].
        coordinate_format (str): Either 'xy' (default) or 'tyx'.

    Returns:
        torch.Tensor: The transformed coordinates with the same shape as `coords`.

    Raises:
        ValueError: If grid sizes don't match the expected lengths for the given coordinate format,
                    or if frame counts (for 'tyx') differ.
    """
    # Convert grid sizes to torch tensors with the same dtype and device as coords.
    if isinstance(input_grid_size, (tuple, list)):
        input_grid_size = torch.tensor(input_grid_size, dtype=coords.dtype, device=coords.device)
    if isinstance(output_grid_size, (tuple, list)):
        output_grid_size = torch.tensor(output_grid_size, dtype=coords.dtype, device=coords.device)
    
    # Validate the grid sizes based on coordinate_format.
    if coordinate_format == 'xy':
        if input_grid_size.numel() != 2 or output_grid_size.numel() != 2:
            raise ValueError("For 'xy' format, grid sizes must have 2 elements.")
    elif coordinate_format == 'tyx':
        if input_grid_size.numel() != 3 or output_grid_size.numel() != 3:
            raise ValueError("For 'tyx' format, grid sizes must have 3 elements.")
        if input_grid_size[0] != output_grid_size[0]:
            raise ValueError("Converting frame count is not supported.")
    else:
        raise ValueError("Recognized coordinate formats are 'xy' and 'tyx'.")
    
    # Compute the transformed coordinates.
    # Broadcasting will apply elementwise division and multiplication.
    transformed_coords = coords * (output_grid_size / input_grid_size)
    
    return transformed_coords


def save_frames_to_mp4(frames, output_path, fps=24, codec='mp4v'):
    """
    Save a list of PIL.Image frames as an MP4 video.

    Args:
        frames (List[PIL.Image.Image]): List of PIL Image frames.
        output_path (str): Path to the output .mp4 file.
        fps (int, optional): Frames per second. Defaults to 24.
        codec (str, optional): FourCC codec code (e.g., 'mp4v', 'H264'). Defaults to 'mp4v'.

    Raises:
        ValueError: If `frames` is empty.
    """
    if not frames:
        raise ValueError("No frames to save.")

    # Ensure all frames are the same size
    width, height = frames[0].size

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in frames:
        # Resize if needed
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)

        # Convert PIL Image (RGB) to BGR array for OpenCV
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()


def save_yaml(
    data: Any,
    filename: str,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False
) -> None:
    """
    Save a Python object to a YAML file. 

    If the file already exists, appends the data as a new YAML document
    (with a leading '---' separator). Otherwise creates a fresh file.

    Args:
        data: The Python object (e.g., dict, list) to serialize.
        filename: Path to the output .yaml file.
        default_flow_style: If False (the default), uses block style.
        sort_keys: If True, sorts dictionary keys in the output.
    """
    # choose append mode if file exists
    mode = 'w'
    with open(filename, mode, encoding='utf-8') as f:
        if mode == 'a':
            # separate from prior content and start a new document
            f.write('\n')
        yaml.safe_dump(
            data,
            f,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=True,
            explicit_start=True
        )


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="V2V motion transfer."
    )
    parser.add_argument("--source_folder", help="Input path to video files", type=str)
    parser.add_argument("--save_folder", help="Output path", type=str)
    parser.add_argument("--num_points", help="Number of tracking points", default=40, type=int)
    args = parser.parse_args()

    n_points = args.num_points
    source_video_folder = args.source_folder
    save_loc = args.save_folder

    os.makedirs(os.path.join(save_loc, 'tracks'), exist_ok=True)
    os.makedirs(os.path.join(save_loc, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(save_loc, 'images'), exist_ok=True)

    model_ = TrackAnyPoint(n_points=n_points)

    t_ll = 121

    kk = 0
    out_list = []

    for fl in os.listdir(source_video_folder):
        frames = load_video_to_frames(os.path.join(source_video_folder, fl))

        frames = frames + [frames[-1]]

        f_len = len(frames)

        print('Processing:', fl)

        for ttt in range(f_len // t_ll):
            if ttt > 0:
                continue
            images = frames[ttt * t_ll:(1 + ttt) * t_ll]

            save_frames_to_mp4(images, os.path.join(save_loc, 'videos', f'{kk}.mp4'))

            image = np.array(images[0])
            images[0].save(os.path.join(save_loc, 'images', f'{kk}.png'))

            caption = ''
            tracks = model_(images)
            tracks = np.transpose(tracks, (2, 1, 0, 3))
            tracks_bytes = array_to_npz_bytes(tracks, os.path.join(save_loc, 'tracks', f'{kk}.pth'), compressed=True)

            out_list.append(
                {
                    'track': os.path.join(save_loc, 'tracks', f'{kk}.pth'),
                    'text': caption,
                    'image': os.path.join(save_loc, 'images', f'{kk}.png'),
                }
            )
            kk += 1

    save_yaml(out_list, os.path.join(save_loc, 'test.yaml'))