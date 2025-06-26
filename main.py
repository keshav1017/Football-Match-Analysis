import cv2
import numpy as np

from trackers.tracker import Tracker
from utils.video_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner


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

    # interpolate ball positions
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # assign ball aquisition
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_info = tracks["ball"][frame_num].get(1)
        if ball_info is None:
            continue  # Skip if ball not detected in this frame

        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True

    # Draw the annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks) 

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
