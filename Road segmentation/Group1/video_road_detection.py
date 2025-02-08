import cv2


from hybridnets import HybridNets, optimized_model

# Initialize video
cap = cv2.VideoCapture('V1.mp4')
frame_rate = 10 # replace with your frame
dim = (1920,1080) # replace with dimesnsion of pics

start_time = 0 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*frame_rate)

# Initialize road detector
model_path = "models/hybridnets_384x640/hybridnets_384x640.onnx"
anchor_path = "models/hybridnets_384x640/anchors_3384x640.npy"
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

out = cv2.VideoWriter('V1_road_detected.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         frame_rate, dim)
cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

	try:
		# Read frame from the video
		ret, new_frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Update road detector
	seg_map, _, _ = roadEstimator(new_frame)

	combined_img = roadEstimator.draw_2D(new_frame)

	cv2.imshow("Road Detections", combined_img)
	out.write(combined_img)



out.release()

    
# Closes all the frames 
cv2.destroyAllWindows()