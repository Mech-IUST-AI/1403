import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables
roi_points = []
selecting = True
roi_segments = []

def select_roi(event):
    """Matplotlib click event to select ROI points."""
    global roi_points, selecting
    if event.button == 1:  # Left-click to add points
        roi_points.append((int(event.xdata), int(event.ydata)))
        plt.scatter(event.xdata, event.ydata, c='red')
        plt.draw()
    elif event.button == 3:  # Right-click to finalize ROI
        selecting = False
        plt.close()

def draw_steering_wheel(frame, direction):
    """Draw a steering wheel icon to indicate direction."""
    center = (100, 100)  # Position of the steering wheel
    radius = 50  # Radius of the steering wheel
    thickness = 5  # Thickness of the steering wheel circle

    # Draw the outer circle of the steering wheel
    cv2.circle(frame, center, radius, (255, 255, 255), thickness)

    # Draw the inner line to indicate direction
    if direction == "Left":
        cv2.line(frame, (center[0], center[1] - radius), (center[0] - radius, center[1]), (0, 255, 0), thickness)
    elif direction == "Right":
        cv2.line(frame, (center[0], center[1] - radius), (center[0] + radius, center[1]), (0, 0, 255), thickness)
    elif direction == "Straight":
        cv2.line(frame, (center[0], center[1] - radius), (center[0], center[1] + radius), (255, 255, 0), thickness)

def process_video_with_roi(video_path, output_path, roi_segments):
    # Define HSV color range for pink
    lower_pink = np.array([140, 50, 50])  # Adjust for your specific pink shade
    upper_pink = np.array([170, 255, 255])

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    all_frame_points = []  # To store points for each frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Find the current ROI for this frame
        current_roi = None
        for segment in roi_segments:
            start_frame, end_frame, roi_vertices = segment
            if start_frame <= frame_number <= end_frame:
                current_roi = roi_vertices
                break

        if current_roi is None:
            print(f"No ROI defined for frame {frame_number}. Skipping frame.")
            continue

        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for pink color
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Create a blank mask for the ROI
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, [np.array(current_roi, np.int32)], 255)

        # Apply the ROI mask to the pink mask
        mask = cv2.bitwise_and(mask, roi_mask)

        # Find contours to extract connected components (road lines)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_points = []  # Points for the current frame
        for contour in contours:
            # Extract points from the contour
            contour_points = contour.squeeze(axis=1).tolist()
            frame_points.extend(contour_points)

            # Draw the contour points for visualization
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        all_frame_points.append(frame_points)  # Store points for the current frame

        # Calculate the minimum and maximum height points and the slope
        direction = "Unknown"
        if frame_points:
            min_point = min(frame_points, key=lambda p: p[1])  # Point with minimum y (highest point)
            max_point = max(frame_points, key=lambda p: p[1])  # Point with maximum y (lowest point)

            # Calculate slope (dy/dx)
            if max_point[0] != min_point[0]:  # Avoid division by zero
                slope = (max_point[1] - min_point[1]) / (max_point[0] - min_point[0])
            else:
                slope = float('inf')  # Vertical line

            # Determine direction based on slope
            if slope > -0.90:
                direction = "Left"
            elif slope < -0.90:
                direction = "Straight"
            else:
                direction = "Right"

            # Draw the min and max points and the slope on the frame
            cv2.circle(frame, tuple(min_point), 5, (255, 0, 0), -1)  # Blue circle for min point
            cv2.circle(frame, tuple(max_point), 5, (0, 0, 255), -1)  # Red circle for max point
           # cv2.putText(frame, f"Slope: {slope:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw the steering wheel to indicate direction
        draw_steering_wheel(frame, direction)

        # Write the processed frame to the output video
        out.write(frame)

        # Print debug information
        print(f"Frame {frame_number}: {len(frame_points)} points detected")
        if frame_points:
            print(f"Min point: {min_point}, Max point: {max_point}, Slope: {slope:.2f}, Direction: {direction}")

    cap.release()
    out.release()

    # Return all points for all frames
    return all_frame_points

# Main script
video_path = "V1_road_detected.avi"  # Replace with your video path
output_path = "V1_lane_detected.avi"

# Open the video to allow frame navigation
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to read the video file.")
    cap.release()
    exit()

# Allow user to define multiple ROIs for different frame ranges
while True:
    # Ask the user for the start frame
    start_frame = int(input("Enter the start frame for this ROI: "))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)  # Set the video to the selected start frame
    ret, frame = cap.read()

    if not ret:
        print(f"Failed to retrieve frame {start_frame}. Please choose a valid frame.")
        continue

    # Convert the selected frame to RGB for Matplotlib display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the selected frame using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)
    ax.set_title("Left-click to select ROI points, right-click to finalize")
    fig.canvas.mpl_connect('button_press_event', select_roi)

    # Wait for ROI selection
    plt.show()

    if len(roi_points) < 3:
        print("You must select at least three points to define a polygon.")
        continue

    # Ask the user for the end frame
    end_frame = int(input("Enter the end frame for this ROI: "))

    roi_segments.append((start_frame, end_frame, roi_points.copy()))
    roi_points = []  # Clear for next ROI

    another = input("Do you want to define another ROI? (yes/no): ").strip().lower()
    if another != 'yes':
        break

print(f"Defined ROIs: {roi_segments}")

# Process the video with the defined ROIs
all_points = process_video_with_roi(video_path, output_path, roi_segments)

# Save all points to a file for further analysis
with open("all_points.txt", "w") as f:
    for frame_idx, points in enumerate(all_points):
        f.write(f"Frame {frame_idx + 1}: {len(points)} points\n")
        f.write(f"{points}\n")

print("Processing complete. Output saved to:", output_path)
print("All points saved to: all_points.txt")