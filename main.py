from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
import cv2
import numpy as np

def main():
    # Read Video
    video_frames = read_video('input_videos/15sec_input_720p.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Draw the annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()