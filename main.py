import cv2
import numpy as np

from trackers.tracker import Tracker
from utils.video_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator

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

    # get object position
    tracker.add_position_to_tracks(tracks)

    # camera  movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_track(tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball positions
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

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

    # Draw camera movement 
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # draw speed and distance 
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, "./output_videos/output_video.avi")

if __name__ == '__main__':
    main()
