from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import math
import cvzone
import json
import pandas as pd

from ByteTrack.byte_tracker import BYTETracker, STrack
import settings

from SORT.sort import *

import streamlit as st
import tempfile



def load_model(model_path):
    model = YOLO(model_path)
    return model


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=1.0, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=10.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    
    return parser


def tracker_options():
    tracker_type = st.radio("Select Tracker", ('ByteTrack', 'SORT'))
    return tracker_type


def display_ROI(image, lanes_coordinates):
    for lane_label, coordinates in lanes_coordinates.items():
        if 'in' in coordinates and 'out' in coordinates:
            cv2.polylines(image, [coordinates['in']], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(image, [coordinates['out']], isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.putText(image, f'{lane_label}', tuple(coordinates['out'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif 'single' in coordinates:
            cv2.polylines(image, [coordinates['single']], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.putText(image, f'{lane_label}', tuple(coordinates['single'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)



def display_vehicle_count(image, zone_counters):
    y_position = 25
    step = 35

    for lane, directions in zone_counters.items():
        if "in" in directions and "out" in directions:
            text = f'LANE {lane} Vehicles IN = {len(directions["in"])} OUT = {len(directions["out"])}'
        elif "single" in directions:
            text = f'LANE {lane} Vehicles = {len(directions["single"])}'
        else:
            continue
        
        cvzone.putTextRect(image, text, [0, y_position], thickness=2, scale=1.5, border=2)
        y_position += step


def convert_to_np_array(path_str):
    path_list = json.loads(path_str.replace("'", '"'))
    points = [(int(point[1]), int(point[2])) for point in path_list if len(point) == 3]
    return np.array(points)


# def process_coord_dataframe(coord_df):
#     coord_df = pd.DataFrame(coord_df)
#     coord_df['path'] = coord_df['path'].apply(convert_to_np_array)
    
#     lanes = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
#     directions = ['in', 'out', 'in', 'out', 'in', 'out', 'in', 'out']

#     coord_df['lane'] = lanes
#     coord_df['direction'] = directions

#     lanes_coordinates = {}
#     for index, row in coord_df.iterrows():
#         lane = row['lane']
#         direction = row['direction']
#         if lane not in lanes_coordinates:
#             lanes_coordinates[lane] = {}
#         lanes_coordinates[lane][direction] = row['path']

#     return lanes_coordinates

def process_coord_dataframe(coord_df):
    coord_df = pd.DataFrame(coord_df)
    coord_df['path'] = coord_df['path'].apply(convert_to_np_array)
    
    lanes_coordinates = {}
    lane_roi_count = {}
    
    for index, row in coord_df.iterrows():
        lane = chr(65 + (index // 2))
        if lane not in lane_roi_count:
            lane_roi_count[lane] = 0
        lane_roi_count[lane] += 1

    for index, row in coord_df.iterrows():
        lane = chr(65 + (index // 2))
        if lane_roi_count[lane] > 1:
            direction = row.get('direction', 'in' if index % 2 == 0 else 'out')
        else:
            direction = 'single'
        if lane not in lanes_coordinates:
            lanes_coordinates[lane] = {}
        lanes_coordinates[lane][direction] = row['path']

    return lanes_coordinates


def _display_detected_frames(conf, model, st_frame, image, model_type, coordinates, tracker_type='ByteTrack'):
    classnames = ['bus', 'car', 'truck']
    width, height = 1366, 768
    half_height = height // 2
        
    lanes_coordinates = coordinates

    image = cv2.resize(image, (width, height))
    half_image = image[half_height:, :]
    detections = np.empty([0, 6])
    
    results = model.predict(source=image, stream=True, conf=conf, verbose=False)
    
    if 'zone_counters' not in st.session_state:
        st.session_state.zone_counters = {}
        for lane, directions in lanes_coordinates.items():
            if 'in' in directions and 'out' in directions:
                st.session_state.zone_counters[lane] = {'in': [], 'out': []}
            elif 'single' in directions:
                st.session_state.zone_counters[lane] = {'single': []}
            else:
                st.session_state.zone_counters[lane] = {}
    zone_counters = st.session_state.zone_counters

    if 'tracker' not in st.session_state:
        if tracker_type == 'ByteTrack':
            args = make_parser().parse_args()
            STrack.reset_next_id()
            st.session_state.tracker = BYTETracker(args, frame_rate=30)
        elif tracker_type == 'SORT':
            KalmanBoxTracker.reset_count()
            st.session_state.tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
    tracker = st.session_state.tracker

    if 'track_history' not in st.session_state:
        st.session_state.track_history = {}
    track_history = st.session_state.track_history

    if model_type == 'Tracking':
        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                confidence = math.ceil(box.conf[0] * 100)
                class_detect = int(box.cls[0])
                class_detect_name = classnames[class_detect]
                if class_detect_name == 'car':
                    class_detect_name = 'c'
                elif class_detect_name == 'truck':
                    class_detect_name = 't'
                else:
                    class_detect_name = 'b'

                new_detections = np.array([x1, y1, x2, y2, confidence, class_detect])
                detections = np.vstack([detections, new_detections])

                cv2.putText(image, f'{class_detect_name}/{confidence}', [x1, y1 - 5], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
        
        if tracker_type == 'ByteTrack':
            track_results = tracker.update(detections, img_info=image.shape, img_size=image.shape)
            
            for track in track_results:
                track_id = track.track_id
                bbox = track.tlbr
                startX, startY, endX, endY = bbox
                startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
                w, h = endX - startX, endY - startY
                cx, cy = startX + w // 2, startY + h // 2
                
                # for lane in lanes_coordinates:
                #     zone_in = lanes_coordinates[lane]['in']
                #     zone_out = lanes_coordinates[lane]['out']

                #     if cv2.pointPolygonTest(zone_in, (cx, cy+height/2), measureDist=False) == 1:
                #         if track_id not in zone_counters[lane]['in']:
                #             zone_counters[lane]['in'].append(track_id)

                #     if cv2.pointPolygonTest(zone_out, (cx, cy+height/2), measureDist=False) == 1:
                #         if track_id not in zone_counters[lane]['out']:
                #             zone_counters[lane]['out'].append(track_id)
                
                for lane in lanes_coordinates:
                    if 'in' in lanes_coordinates[lane] and 'out' in lanes_coordinates[lane]:
                        zone_in = lanes_coordinates[lane]['in']
                        zone_out = lanes_coordinates[lane]['out']

                        if cv2.pointPolygonTest(zone_in, (cx, cy), measureDist=False) == 1:
                            if track_id not in zone_counters[lane]['in']:
                                zone_counters[lane]['in'].append(track_id)

                        if cv2.pointPolygonTest(zone_out, (cx, cy), measureDist=False) == 1:
                            if track_id not in zone_counters[lane]['out']:
                                zone_counters[lane]['out'].append(track_id)

                    elif 'single' in lanes_coordinates[lane]:
                        zone_single = lanes_coordinates[lane]['single']

                        if cv2.pointPolygonTest(zone_single, (cx, cy ), measureDist=False) == 1:
                            if track_id not in zone_counters[lane]['single']:
                                zone_counters[lane]['single'].append(track_id)
                
                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 15:
                    track_history[track_id].pop(0)

                for i in range(1, len(track_history[track_id])):
                    cv2.line(image, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 0), 2)

                cv2.putText(image, f'{track_id}', (endX - 15, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

        elif tracker_type == 'SORT':
            track_results = tracker.update(detections)
            for result in track_results:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                '''For drawing vehicles lines'''
                if id not in track_history:
                    track_history[id] = []

                track_history[id].append((cx, cy))
                if len(track_history[id]) > 10:
                    track_history[id].pop(0)

                for i in range(1, len(track_history[id])):
                    cv2.line(image, track_history[id][i - 1], track_history[id][i], (0, 255, 0), 2)

                for lane in lanes_coordinates:
                    if 'in' in lanes_coordinates[lane] and 'out' in lanes_coordinates[lane]:
                        zone_in = lanes_coordinates[lane]['in']
                        zone_out = lanes_coordinates[lane]['out']

                        if cv2.pointPolygonTest(zone_in, (cx, cy), measureDist=False) == 1:
                            if track_id not in zone_counters[lane]['in']:
                                zone_counters[lane]['in'].append(track_id)

                        if cv2.pointPolygonTest(zone_out, (cx, cy), measureDist=False) == 1:
                            if track_id not in zone_counters[lane]['out']:
                                zone_counters[lane]['out'].append(track_id)

                    elif 'single' in lanes_coordinates[lane]:
                        zone_single = lanes_coordinates[lane]['single']

                        if cv2.pointPolygonTest(zone_single, (cx, cy ), measureDist=False) == 1:
                            if track_id not in zone_counters[lane]['single']:
                                zone_counters[lane]['single'].append(track_id)
                
                cv2.putText(image, f'{id}', (x2 - 15, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
        
        display_vehicle_count(image, zone_counters)

        display_ROI(image, lanes_coordinates)
        
    elif model_type == 'Detection':
        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                w, h = x2 - x1, y2 - y1
                confidence = math.ceil(box.conf[0] * 100)
                class_detect = int(box.cls[0])
                class_detect_name = classnames[class_detect]
                if class_detect_name == 'car':
                    class_detect_name = 'c'
                elif class_detect_name == 'truck':
                    class_detect_name = 't'
                else:
                    class_detect_name = 'b'
                
                new_detections = np.array([x1, y1, x2, y2, confidence, class_detect])
                detections = np.vstack([detections, new_detections])

                cv2.putText(image, f'{class_detect_name}/{confidence}', [x1, y1 - 5], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    st_frame.image(image,
                caption='Detected Video',
                channels="BGR",
                use_column_width=True
                )


# def play_webcam(conf, model, model_type):
#     source_webcam = settings.WEBCAM_PATH
#     if st.sidebar.button('Detect Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(source_webcam)
#             st.session_state.reset_tracker = True
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if not success:
#                     vid_cap.release()
#                     break
#                 _display_detected_frames(conf,
#                                         model,
#                                         st_frame,
#                                         image,
#                                         model_type,
#                                         )
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))


def play_selected_video(conf, model, model_type):
    coordinates = None   

    tracker_type = tracker_options()

    source_vid = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    # upload_coord = st.sidebar.checkbox("Upload Zone Coordinates")
    
    # if not upload_coord:
    #     try:
    #         coordinates = st.session_state['json']
    #     except Exception as e:
    #         st.write()
    # else:
    #     coordinates = st.sidebar.file_uploader(
    #         label = "Upload counting zone coordinates"
    #     )
    if model_type == 'Tracking':
        coordinates_file = st.sidebar.file_uploader(
            label = "Upload counting zone coordinates"
        )

        if coordinates_file is not None: 
            coordinates = pd.read_csv(coordinates_file)
            coordinates.drop(columns=coordinates.columns[0], axis=1, inplace=True)
            coordinates = process_coord_dataframe(coordinates)
        else:
            try:
                coordinates = st.session_state['json']
                coordinates = process_coord_dataframe(coordinates)
            except Exception as e:
                st.write("ROI (Region of Interest) Required")

    if source_vid:
        st.video(source_vid)

    if st.sidebar.button('Detect Video Objects'):
        if ("tracker" or "zone_counters" or "track_history") in st.session_state:
            del st.session_state.zone_counters
            del st.session_state.tracker
            del st.session_state.track_history
        try:
            tfile = tempfile.NamedTemporaryFile()
            tfile.write(source_vid.read())
            vid_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if not success:
                    vid_cap.release()
                    break
                _display_detected_frames(conf,
                                        model,
                                        st_frame,
                                        image,
                                        model_type,
                                        coordinates,
                                        tracker_type
                                        )

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))