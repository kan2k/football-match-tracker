from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video("input_videos/17.mp4")
    tracker = Tracker()
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/tracks.pkl"
    )
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, "output.avi")


if __name__ == "__main__":
    main()
