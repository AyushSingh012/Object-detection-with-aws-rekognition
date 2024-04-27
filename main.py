import os
import boto3
import cv2
import credentials

# Output directories
output_dir = './data'
output_dir_imgs = os.path.join(output_dir, 'imgs')
output_dir_anns = os.path.join(output_dir, 'anns')
output_dir_detected = os.path.join(output_dir, 'detected_frames')
output_dir_video = './'

# Create output directories if they don't exist
os.makedirs(output_dir_imgs, exist_ok=True)
os.makedirs(output_dir_anns, exist_ok=True)
os.makedirs(output_dir_detected, exist_ok=True)

# Create AWS Reko client
reko_client = boto3.client('rekognition',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key,
                           region_name='ap-southeast-2'
                           )

# Set the target class
target_class = 'Zebra'

# Load Video
cap = cv2.VideoCapture('./zebras.mp4')
frame_nmr = -1

# Read Frames
ret = True
while ret:
    ret, frame = cap.read()

    if ret:
        frame_nmr += 1
        H, W, _ = frame.shape

        # Convert frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert buffer to bytes
        image_bytes = buffer.tobytes()

        # Detect Objects (Using Amazon Rekognition)
        response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                              MinConfidence=50)

        # Save all frames to imgs folder
        cv2.imwrite(os.path.join(output_dir_imgs, 'frame_{}.jpg'.format(str(frame_nmr).zfill(6))), frame)

        # Write annotations
        with open(os.path.join(output_dir_anns, 'frame_{}.txt'.format(str(frame_nmr).zfill(6))), 'w') as f:
            for label in response['Labels']:
                for instance in label.get('Instances', []):
                    bbox = instance['BoundingBox']
                    x1 = bbox['Left']
                    y1 = bbox['Top']
                    width = bbox['Width']
                    height = bbox['Height']
                    f.write(f"{label['Name']} {x1} {y1} {width} {height}\n")

        # Save frames with detected objects and bounding boxes to detected_frames folder
        frame_with_boxes = frame.copy()
        for label in response['Labels']:
            if label['Name'] == target_class:
                for instance in label.get('Instances', []):
                    bbox = instance['BoundingBox']
                    x1 = int(bbox['Left'] * W)
                    y1 = int(bbox['Top'] * H)
                    width = int(bbox['Width'] * W)
                    height = int(bbox['Height'] * H)
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(output_dir_detected, 'frame_{}.jpg'.format(str(frame_nmr).zfill(6))), frame_with_boxes)

# Release VideoCapture
cap.release()

# Create video from frames with detected objects and bounding boxes
output_video_path = 'detected_objects_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(output_dir_video, output_video_path), fourcc, 30, (W, H))
for i in range(frame_nmr + 1):
    frame = cv2.imread(os.path.join(output_dir_detected, f"frame_{str(i).zfill(6)}.jpg"))
    out.write(frame)
out.release()
