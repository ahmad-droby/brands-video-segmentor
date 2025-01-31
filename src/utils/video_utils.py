import os
from typing import List, Tuple, Dict
from moviepy.editor import VideoFileClip

from config.settings import SEGMENT_PADDING
from utils.logger import get_logger

logger = get_logger()

def group_detections_by_time(
    detections: List[Tuple[str, float]],
    max_gap: float = 1.0
) -> Dict[str, List[Tuple[float, float]]]:
    """
    detections: list of (brand, time_in_seconds).
    max_gap: maximum gap in seconds to consider consecutive frames as one segment.

    Returns a dictionary: { brand: [(start, end), (start, end), ...] }.
    """
    # Organize times per brand
    brand_times = {}
    for brand, t in detections:
        brand_times.setdefault(brand, []).append(t)

    # For each brand, group contiguous times
    brand_segments = {}
    for brand, times in brand_times.items():
        times.sort()
        segments = []
        seg_start = times[0]
        seg_end = seg_start

        for current_t in times[1:]:
            if current_t - seg_end <= max_gap:
                seg_end = current_t
            else:
                segments.append((seg_start, seg_end))
                seg_start = current_t
                seg_end = current_t
        # final
        segments.append((seg_start, seg_end))
        brand_segments[brand] = segments

    return brand_segments

def cut_video_segments(
    video_path: str,
    brand_segments: Dict[str, List[Tuple[float, float]]],
    output_dir: str,
    padding: float = SEGMENT_PADDING
):
    """
    Cuts the video around brand segments + 'padding' seconds before and after.
    Each brand's segments are saved as separate clips in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)

    for brand, segments in brand_segments.items():
        for i, (start, end) in enumerate(segments):
            segment_start = max(start - padding, 0)
            segment_end = min(end + padding, clip.duration)
            sub_clip = clip.subclip(segment_start, segment_end)

            output_name = f"{brand}_{int(segment_start)}_{int(segment_end)}.mp4"
            out_path = os.path.join(output_dir, output_name)
            logger.info(f"Saving clip for brand='{brand}' from {segment_start}s to {segment_end}s -> {out_path}")

            sub_clip.write_videofile(out_path, codec="libx264", audio_codec="aac")