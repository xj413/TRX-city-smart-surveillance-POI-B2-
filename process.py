import os
import subprocess

source_video = "/Users/user/Downloads/raw_test_bag_20260818_101233/filtered videos/firehose.mp4"
model_path = "models/firehose_cabinet_nano_v2.onnx"
output_video = "annotated.mp4"
output_gif = "demo.gif"

print("Running detection...")
ret1 = os.system(f'python3 detect.py --model "{model_path}" --classes "hose_closed,hose_open" --source "{source_video}" --out "{output_video}"')
if ret1 != 0:
    print("Detection failed!")
    exit(1)

print("Converting to GIF...")
ret2 = os.system(f'ffmpeg -i "{output_video}" -vf "fps=10,scale=320:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize "{output_gif}"')
if ret2 != 0:
    # Try alternative ffmpeg command
    ret3 = os.system(f'ffmpeg -i "{output_video}" -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "{output_gif}"')
    if ret3 != 0:
        print("GIF conversion failed!")
        exit(1)

print("Done!")
