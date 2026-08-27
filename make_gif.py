import cv2
from PIL import Image
from detect import BinFullnessDetector

source_video = "/Users/user/Downloads/raw_test_bag_20260818_101233/filtered videos/firehose.mp4"
model_path = "models/firehose_cabinet_nano_v2.onnx"
output_gif = "demo.gif"

print("Loading model...")
det = BinFullnessDetector(model_path, class_names=("hose_closed", "hose_open"), conf_thres=0.35, iou_thres=0.45)

print("Opening video...")
cap = cv2.VideoCapture(source_video)
frames = []
count = 0

# Just grab 30 frames to make a small GIF
while count < 30:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process every 5th frame to make the GIF cover more time and stay small
    if count % 5 == 0:
        dets = det(frame)
        vis = det.draw(frame, dets)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        # Resize to make GIF smaller
        vis_rgb = cv2.resize(vis_rgb, (320, int(320 * vis_rgb.shape[0] / vis_rgb.shape[1])))
        frames.append(Image.fromarray(vis_rgb))
    
    count += 1

cap.release()

if frames:
    print("Saving GIF...")
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print("Successfully created demo.gif")
else:
    print("No frames processed!")
