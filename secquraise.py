import cv2
import numpy as np

COLOR_RANGES = {
    "red": ([0, 120, 70], [10, 255, 255]),
    "orange": ([10, 100, 20], [25, 255, 255]),
    "green": ([35, 100, 100], [85, 255, 255]),
    "yellow": ([25, 150, 150], [35, 255, 255]),
    "blue": ([100, 150, 0], [140, 255, 255]),
    "white": ([0, 0, 200], [180, 20, 255]),
    "gray": ([0, 0, 50], [180, 30, 220])
}

def track_balls(video_path, output_video_path, output_log_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))
    
    event_log = []

    frame_count = 0
    ball_positions = {color: None for color in COLOR_RANGES.keys()}
    ball_quadrants = {color: None for color in COLOR_RANGES.keys()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / 20.0  

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color, (lower, upper) in COLOR_RANGES.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.GaussianBlur(mask, (9, 9), 2, 2)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                for c in contours:
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    else:
                        center = (0, 0)

                    if radius > 20 and radius < 100:
                        perimeter = cv2.arcLength(c, True)
                        area = cv2.contourArea(c)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)

                        if 0.7 < circularity <= 1.2: 
                            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                            cv2.putText(frame, color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            current_quadrant = None 
                            if ball_positions[color] is not None:
                                prev_x, prev_y = ball_positions[color]
                                prev_quadrant = None

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
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    with open(output_log_path, 'w') as f:
        for event in event_log:
            f.write(event + "\n")

if __name__ == "__main__":
    video_path = r"C:\Users\aarya\Downloads\AI Assignment video.mp4"
    output_video_path = "output_video.avi"
    output_log_path = "event_log.txt"
    
    track_balls(video_path, output_video_path, output_log_path)
    
    print(f"Processed video saved to {output_video_path}")
    print(f"Event log saved to {output_log_path}")
