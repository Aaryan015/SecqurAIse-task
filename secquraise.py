import cv2
import numpy as np
import time
import os

BALL_COLORS = {
    "red": ([0, 0, 100], [50, 50, 255]),
    "green": ([0, 100, 0], [50, 255, 50]),
    "blue": ([100, 0, 0], [255, 50, 50])
}

QUADRANTS = {
    1: (0, 0, 320, 240),
    2: (320, 0, 640, 240),
    3: (0, 240, 320, 480),
    4: (320, 240, 640, 480)
}

def get_quadrant(x, y):
    for q_num, (x1, y1, x2, y2) in QUADRANTS.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return q_num
    return None

def track_balls(video_path, output_video_path, output_log_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))
    
    event_log = []

    frame_count = 0
    ball_positions = {color: None for color in BALL_COLORS.keys()}
    ball_quadrants = {color: None for color in BALL_COLORS.keys()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / 20.0  

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color, (lower, upper) in BALL_COLORS.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 10:
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.putText(frame, color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    current_quadrant = get_quadrant(center[0], center[1])
                    if ball_positions[color] is not None:
                        prev_x, prev_y = ball_positions[color]
                        prev_quadrant = get_quadrant(prev_x, prev_y)

                        if current_quadrant != prev_quadrant:
                            if current_quadrant is not None:
                                event_log.append(f"{timestamp}, {current_quadrant}, {color}, Entry")
                                cv2.putText(frame, f"Entry {color}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            if prev_quadrant is not None:
                                event_log.append(f"{timestamp}, {prev_quadrant}, {color}, Exit")
                                cv2.putText(frame, f"Exit {color}", (prev_x, prev_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    ball_positions[color] = center
                    ball_quadrants[color] = current_quadrant
        
        out.write(frame)

    cap.release()
    out.release()
    
    with open(output_log_path, 'w') as f:
        for event in event_log:
            f.write(event + "\n")

if __name__ == "__main__":
    video_path = "C:\\Users\\aarya\\Downloads\\AI Assignment video.mp4"
    output_video_path = "C:\\Users\\aarya\\Downloads\\output_video.avi"
    output_log_path = "C:\\Users\\aarya\\Downloads\\event_log.txt"
    
    track_balls(video_path, output_video_path, output_log_path)
    
    print(f"Processed video saved to {output_video_path}")
    print(f"Event log saved to {output_log_path}")