from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import cv2
import os


class Tracker:
    def __init__(self):
        self.model = YOLO("models/best.pt")
        self.tracker = sv.ByteTrack()

    def detect_frames(self, video_frames):
        detections = self.model.predict(video_frames, conf=0.25)
        return detections

    def get_object_tracks(self, video_frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(video_frames)

        tracks = {
            "player": [],
            "referee": [],
            "ball": [],
        }

        for frame_num, detections in enumerate(detections):
            cls_names = detections.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detections)

            # goalkeeper doesnt work with tracking
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # tracking
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )
            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            # player and referee tracking
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                elif cls_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            # ball tracking
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_rectangle(self, frame, bbox, color, track_id):
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_4)

        if track_id:
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame

    def draw_arrow(self, frame, start_point, end_point, color):
        cv2.arrowedLine(frame, start_point, end_point, color, 2, tipLength=0.3)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            ball_center = None
            if ball_dict and 1 in ball_dict:
                ball_center = get_center_of_bbox(ball_dict[1]["bbox"])

            player_distances = []
            if ball_center:
                for track_id, player in player_dict.items():
                    player_center = get_center_of_bbox(player["bbox"])
                    distance = np.sqrt(
                        (ball_center[0] - player_center[0]) ** 2
                        + (ball_center[1] - player_center[1]) ** 2
                    )
                    player_distances.append((track_id, distance))
                player_distances.sort(key=lambda x: x[1])

            # Draw players with arrows for 5 nearest to ball
            for track_id, player in player_dict.items():
                frame = self.draw_rectangle(
                    frame, player["bbox"], (255, 0, 0), track_id
                )

                if ball_center:
                    closest_players = [p[0] for p in player_distances[:5]]
                    if track_id in closest_players:
                        player_center = get_center_of_bbox(player["bbox"])
                        frame = self.draw_arrow(
                            frame, player_center, ball_center, (0, 0, 255)
                        )

            for track_id, referee in referee_dict.items():
                frame = self.draw_rectangle(
                    frame, referee["bbox"], (0, 255, 0), track_id
                )

            if ball_dict and 1 in ball_dict:
                frame = self.draw_rectangle(
                    frame, ball_dict[1]["bbox"], (0, 0, 255), None
                )

            output_video_frames.append(frame)

        return output_video_frames


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
