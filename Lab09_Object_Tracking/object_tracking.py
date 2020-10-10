'''
   Lab09: Object Tracking
'''

'''
How to Run:
object_tracking.py --video dash_cam_video.mp4 --tracker csrt
'''

'''
Press 's' key to select ROI and hit SPACE or ENTER
Press 'q' key to end video stream at any time
'''

# Import Statements
import cv2
import argparse
import sys
from imutils.video import FPS
import imutils

# Setup Tracker
def create_tracker(tracker_type):
	# Extract cv2 version
	(major, minor) = cv2.__version__.split(".")[:2]

	if int(minor) < 3:
		tracker = cv2.Tracker_create(tracker_type.upper())
	
	else:
		OPENCV_OBJECT_TRACKERS = {
			"csrt": cv2.TrackerCSRT_create,
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create
		}

		tracker = OPENCV_OBJECT_TRACKERS[tracker_type.lower()]()

	return tracker

def read_video_and_track(video_file, tracker, tracker_type):
	# Initialize the bbox coordinates to track
	initBB = None
	# Initialize FPS throughput estimator
	fps = None

	print("Press s to select roi and q to end video_stream")
	
	# Reference to video_file
	vs = cv2.VideoCapture(video_file)

	# Loop over frames
	while True:
		# Read current frame
		ok, frame = vs.read()

		# Check if reached end of stream
		if frame is None:
			break

		# Resize frame to process faster
		frame = imutils.resize(frame, width=500)
		H, W = frame.shape[:2]

		# Check if tracking object
		if initBB is not None:
			success, box = tracker.update(frame)

			# Display BBox coordinates on frame
			if success:
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),
					(0, 255, 0), 2)

			# Update FPS counter
			fps.update()
			fps.stop()

			# Initialize info to display
			display_info = [
				("Tracker", tracker_type),
				("Success", "Yes" if success else "No"),
				("FPS", "{:.2f}".format(fps.fps())),
			]

			# Display Information on frame
			for (i, (k, v)) in enumerate(display_info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# Show output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xff

		# If 's' key selected "select" bbox to track
		if key == ord('s'):
			# Select bbox coordinates
			initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

			# Start tracker
			tracker.init(frame, initBB)
			fps = FPS().start()

		# if 'q' key selected end stream
		elif key == ord('q'):
			break

	vs.release()
	cv2.destroyAllWindows()

if __name__ == '__main__' :
	# Parse Command Line Arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", type=str,
		help="path to input video file")
	ap.add_argument("-t", "--tracker", type=str,
		help="OpenCV object tracker type")
	args = vars(ap.parse_args())

	# Create Tracker
	tracker = create_tracker(args['tracker'])

	# Read Video and Track
	read_video_and_track(args['video'], tracker, args['tracker'])