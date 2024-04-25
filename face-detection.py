# import cv2
# import numpy as np
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# from moviepy.editor import ImageSequenceClip


# # Load the input video
# cap = cv2.VideoCapture('VID_shailendra.mp4')
# cap = cv2.VideoCapture('VID_shailendra.mp4')
# if cap is None:
#     print("Error: Could not open video file")
#     exit()
# # Load the Haar Cascade Classifier for face detection
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize variables for tracking the face position
# prev_x, prev_y, prev_w, prev_h = None, None, None, None
# center_x, center_y = None, None
# crop_x, crop_y = None, None

# # Create an empty list to store the cropped frames
# cropped_frames = []

# # Process each frame of the video
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # If a face is detected, use its position to track it
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         center_x, center_y = x + w/2, y + h/2
#         prev_x, prev_y, prev_w, prev_h = x, y, w, h
#     else:
#         # If no face is detected, use the previous position to track it
#         if prev_x is not None:
#             center_x, center_y = prev_x + prev_w/2, prev_y + prev_h/2

#     # If we have a valid face position, calculate the crop position
#     if center_x is not None and center_y is not None:
#         crop_width, crop_height = min(frame.shape[1], frame.shape[0] * 9/16), min(frame.shape[0], frame.shape[1] * 9/16)
#         crop_x = max(0, min(center_x - crop_width/2, frame.shape[1] - crop_width))
#         crop_y = max(0, min(center_y - crop_height/2, frame.shape[0] - crop_height))

#     # Draw a rectangle around the detected face
#     # Display the frame
#     cv2.imshow('frame', frame)

#     # Crop the frame and add it to the list
#     if crop_x is not None and crop_y is not None:
#         crop_frame = frame[int(crop_y):int(crop_y+crop_height), int(crop_x):int(crop_x+crop_width)]
#         cv2.imwrite('/Users/mohamedkhalaf/Downloads/TikTok Project/temp.jpg', crop_frame)
#         cropped_frames.append(VideoFileClip('temp.jpg').resize(height=1080))

#     # Wait for a key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Concatenate the cropped frames to a single video clip
# if cropped_frames:
#     final_clip = concatenate_videoclips(cropped_frames)
#     fps = 24  # Change this to the desired frame rate
#     final_clip.write_videofile('output_video2.mp4', fps= 60, audio=True, verbose=False, codec='libx264', temp_audiofile='temp-audio.m4a', remove_temp=True, ffmpeg_params=['-vcodec', 'libx264', '-profile:v', 'high', '-level:v', '4.0', '-preset', 'slow'])















# import cv2
# import numpy as np
# from moviepy.editor import VideoFileClip, concatenate_videoclips

# # Load the input video
# cap = cv2.VideoCapture('VID_shailendra.mp4')
# if cap is None:
#     print("Error: Could not open video file")
#     exit()
# # Load the Haar Cascade Classifier for face detection
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize variables for tracking the face position
# prev_x, prev_y, prev_w, prev_h = None, None, None, None
# center_x, center_y = None, None
# crop_x, crop_y = None, None

# # Create an empty list to store the cropped frames
# cropped_frames = []

# # Initialize the video writer
# # Initialize the video writer with the 'h264' codec
# # fourcc = cv2.VideoWriter_fourcc(*'h264')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

# # Process each frame of the video
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # If a face is detected, use its position to track it
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         center_x, center_y = x + w/2, y + h/2
#         prev_x, prev_y, prev_w, prev_h = x, y, w, h
#     else:
#         # If no face is detected, use the previous position to track it
#         if prev_x is not None:
#             center_x, center_y = prev_x + prev_w/2, prev_y + prev_h/2

#     # If we have a valid face position, calculate the crop position
#     if center_x is not None and center_y is not None:
#         crop_width, crop_height = min(frame.shape[1], frame.shape[0] * 9/16), min(frame.shape[0], frame.shape[1] * 9/16)
#         crop_x = max(0, min(center_x - crop_width/2, frame.shape[1] - crop_width))
#         crop_y = max(0, min(center_y - crop_height/2, frame.shape[0] - crop_height))

#     # Draw a rectangle around the detected face
#     if prev_x is not None:
#         cv2.rectangle(frame, (prev_x, prev_y), (prev_x + prev_w, prev_y + prev_h), (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('frame', frame)

# # Crop the frame and add it to the list
#     if crop_x is not None and crop_y is not None:
#         crop_frame = frame[int(crop_y):int(crop_y+crop_height), int(crop_x):int(crop_x+crop_width)]
#         cropped_frames.append(crop_frame)
#         out.write(crop_frame)

#         # Update the output video dimensions with the size of the cropped frames
#         out.set(3, crop_frame.shape[1])
#         out.set(4, crop_frame.shape[0])


#     if crop_x is not None and crop_y is not None:
#         crop_frame = frame[int(crop_y):int(crop_y+crop_height), int(crop_x):int(crop_x+crop_width)]
#         cv2.imwrite('/Users/vivek/Documents/face-detection/temp.jpg', crop_frame)
#         cropped_frames.append(VideoFileClip('temp.jpg').resize(height=1080))

#     # Wait for a key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from moviepy.editor import ImageSequenceClip

# # Load the input video
# cap = cv2.VideoCapture('VID_shailendra.mp4')
# if not cap.isOpened():
#     print("Error: Could not open video file")
#     exit()

# # Load the Haar Cascade Classifier for face detection
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize variables for tracking the face position
# prev_x, prev_y, prev_w, prev_h = None, None, None, None
# center_x, center_y = None, None
# crop_x, crop_y = None, None

# # Create an empty list to store the cropped frames
# cropped_frames = []

# # Process each frame of the video
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # If a face is detected, use its position to track it
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         center_x, center_y = x + w/2, y + h/2
#         prev_x, prev_y, prev_w, prev_h = x, y, w, h
#     else:
#         # If no face is detected, use the previous position to track it
#         if prev_x is not None:
#             center_x, center_y = prev_x + prev_w/2, prev_y + prev_h/2

#     # If we have a valid face position, calculate the crop position
#     if center_x is not None and center_y is not None:
#         crop_width, crop_height = min(frame.shape[1], frame.shape[0] * 9/16), min(frame.shape[0], frame.shape[1] * 9/16)
#         crop_x = max(0, min(center_x - crop_width/2, frame.shape[1] - crop_width))
#         crop_y = max(0, min(center_y - crop_height/2, frame.shape[0] - crop_height))

#     # Crop the frame and add it to the list
#     # Crop the frame and add it to the list
#     # if crop_x is not None and crop_y is not None:
#     #     crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
#     #     cropped_frames.append(crop_frame)
#     standard_width, standard_height = 640, 1080

#     if crop_x is not None and crop_y is not None:
#         crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
#         # Resize the frame to the standard size
#         resized_frame = cv2.resize(crop_frame, (standard_width, standard_height))
#         cropped_frames.append(resized_frame)
#     # Wait for a key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Convert the list of frames to a video clip
# if cropped_frames:
#     final_clip = ImageSequenceClip(cropped_frames, fps=24)  # No need to convert color space since frames are already in RGB
#     final_clip.write_videofile('output_video2.mp4', fps=60, codec='libx264')





# import cv2
# import numpy as np
# from moviepy.editor import ImageSequenceClip

# # Load the input video
# cap = cv2.VideoCapture('test-1.mp4')
# if not cap.isOpened():
#     print("Error: Could not open video file")
#     exit()

# # Load the Haar Cascade Classifier for face detection
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize variables for tracking the face position
# prev_x, prev_y, prev_w, prev_h = None, None, None, None
# center_x, center_y = None, None
# crop_x, crop_y = None, None

# # Create an empty list to store the cropped frames
# cropped_frames = []

# # Process each frame of the video
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # If a face is detected, use its position to track it
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         center_x, center_y = x + w/2, y + h/2
#         prev_x, prev_y, prev_w, prev_h = x, y, w, h
#     else:
#         # If no face is detected, use the previous position to track it
#         if prev_x is not None:
#             center_x, center_y = prev_x + prev_w/2, prev_y + prev_h/2

#     # If we have a valid face position, calculate the crop position
#     if center_x is not None and center_y is not None:
#         crop_width, crop_height = min(frame.shape[1], frame.shape[0] * 9/16), min(frame.shape[0], frame.shape[1] * 9/16)
#         crop_x = max(0, min(center_x - crop_width/2, frame.shape[1] - crop_width))
#         crop_y = max(0, min(center_y - crop_height/2, frame.shape[0] - crop_height))

#     # Crop the frame and add it to the list
#     # Crop the frame and add it to the list
#     # if crop_x is not None and crop_y is not None:
#     #     crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
#     #     cropped_frames.append(crop_frame)
#     standard_width, standard_height = 640, 1080
#     if crop_x is not None and crop_y is not None:
#         crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
        
#         # Resize the frame to the standard size
#         resized_frame = cv2.resize(crop_frame, (standard_width, standard_height))
        
#         # Convert color space from BGR to RGB
#         rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
#         # Optional: Enhance color/contrast here if needed

#         cropped_frames.append(rgb_frame)
#     # Wait for a key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Convert the list of frames to a video clip
# if cropped_frames:
#     final_clip = ImageSequenceClip(cropped_frames, fps=24)  # No need to convert color space since frames are already in RGB
#     final_clip.write_videofile('output_video2.mp4', fps=60, codec='libx264')








# import cv2
# import numpy as np
# from moviepy.editor import VideoFileClip, ImageSequenceClip

# # Load the input video
# cap = cv2.VideoCapture('test-1.mp4')
# if not cap.isOpened():
#     print("Error: Could not open video file")
#     exit()

# # Load the Haar Cascade Classifier for face detection
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize variables for tracking the face position
# prev_x, prev_y, prev_w, prev_h = None, None, None, None
# center_x, center_y = None, None
# crop_x, crop_y = None, None

# # Create an empty list to store the cropped frames
# cropped_frames = []

# # Process each frame of the video
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # If a face is detected, use its position to track it
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         center_x, center_y = x + w/2, y + h/2
#         prev_x, prev_y, prev_w, prev_h = x, y, w, h
#     else:
#         # If no face is detected, use the previous position to track it
#         if prev_x is not None:
#             center_x, center_y = prev_x + prev_w/2, prev_y + prev_h/2

#     # If we have a valid face position, calculate the crop position
#     if center_x is not None and center_y is not None:
#         crop_width, crop_height = min(frame.shape[1], frame.shape[0] * 9/16), min(frame.shape[0], frame.shape[1] * 9/16)
#         crop_x = max(0, min(center_x - crop_width/2, frame.shape[1] - crop_width))
#         crop_y = max(0, min(center_y - crop_height/2, frame.shape[0] - crop_height))

#     # Crop the frame and add it to the list
#     # Crop the frame and add it to the list
#     # if crop_x is not None and crop_y is not None:
#     #     crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
#     #     cropped_frames.append(crop_frame)
#     standard_width, standard_height = 640, 1080
#     if crop_x is not None and crop_y is not None:
#         crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
        
#         # Resize the frame to the standard size
#         resized_frame = cv2.resize(crop_frame, (standard_width, standard_height))
        
#         # Convert color space from BGR to RGB
#         rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
#         # Optional: Enhance color/contrast here if needed

#         cropped_frames.append(rgb_frame)
#     # Wait for a key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# # Load the original video using moviepy to extract audio
# original_video = VideoFileClip('test-1.mp4')
# audio = original_video.audio

# # Convert the list of frames to a video clip
# if cropped_frames:
#     final_clip = ImageSequenceClip(cropped_frames, fps=24)  # Assuming the fps is 24
#     # Set the audio of the final clip
#     final_clip = final_clip.set_audio(audio)
#     final_clip.write_videofile('output_video_with_audio.mp4', fps=60, codec='libx264', audio_codec='aac')

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip

# Load the input video
cap = cv2.VideoCapture('aks-3.mp4')
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Load the Haar Cascade Classifier for face detection
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for tracking the face position
prev_x, prev_y, prev_w, prev_h = None, None, None, None
smoothed_center_x, smoothed_center_y = None, None
smoothing_factor = 0.000001*0.000001  # Adjust this value to control the amount of smoothing

# Create an empty list to store the cropped frames
cropped_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = trained_face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        center_x, center_y = x + w/2, y + h/2

        # Smooth the center position
        if smoothed_center_x is None or smoothed_center_y is None:
            smoothed_center_x, smoothed_center_y = center_x, center_y
        else:
            smoothed_center_x = (1 - smoothing_factor) * smoothed_center_x + smoothing_factor * center_x
            smoothed_center_y = (1 - smoothing_factor) * smoothed_center_y + smoothing_factor * center_y

        prev_x, prev_y, prev_w, prev_h = x, y, w, h
    else:
        if prev_x is not None:
            center_x, center_y = prev_x + prev_w/2, prev_y + prev_h/2
            smoothed_center_x, smoothed_center_y = center_x, center_y

    if smoothed_center_x is not None and smoothed_center_y is not None:
        crop_width, crop_height = min(frame.shape[1], frame.shape[0] * 9/16), min(frame.shape[0], frame.shape[1] * 9/16)
        crop_x = max(0, min(smoothed_center_x - crop_width/2, frame.shape[1] - crop_width))
        crop_y = max(0, min(smoothed_center_y - crop_height/2, frame.shape[0] - crop_height))

        standard_width, standard_height = 640, 1080
        crop_frame = frame[int(crop_y):int(crop_y + crop_height), int(crop_x):int(crop_x + crop_width)]
        resized_frame = cv2.resize(crop_frame, (standard_width, standard_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        cropped_frames.append(rgb_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

original_video = VideoFileClip('aks-3.mp4')
audio = original_video.audio

if cropped_frames:
    final_clip = ImageSequenceClip(cropped_frames, fps=25)  # Adjust fps if needed
    final_clip = final_clip.set_audio(audio)
    final_clip.write_videofile('output_video_with_audio.mp4', fps=20, codec='libx264', audio_codec='aac')
