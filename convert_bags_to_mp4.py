import os
from pathlib import Path
import glob
import cv2
import numpy as np
import argparse
import imageio
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores


def encoding_to_cvtype(encoding):
    if encoding in ['bgr8', '8UC3']:
        return cv2.COLOR_BGR2RGB
    elif encoding in ['rgb8']:
        return None  # No conversion needed, imageio expects RGB
    elif encoding in ['mono8', '8UC1']:
        return cv2.COLOR_GRAY2RGB
    elif encoding in ['yuv422', 'yuv422_yuy2']:
        return cv2.COLOR_YUV2RGB_YUY2
    elif encoding in ['nv12']:
        return cv2.COLOR_YUV2RGB_NV12
    else:
        print(f"Warning: Unknown or unhandled encoding '{encoding}'. Returning as is.")
        return None


def process_bag(bag_path, out_dir):
    print(f"Processing bag: {bag_path}")
    bag_name = os.path.basename(os.path.normpath(bag_path))

    # Check if this bag has already been processed (any mp4 starting with bag_name exists)
    existing_mp4s = glob.glob(os.path.join(out_dir, f"{bag_name}*.mp4"))
    if existing_mp4s:
        # We will delete them and re-run since the previous ones were empty/corrupted
        for f in existing_mp4s:
            try:
                os.remove(f)
            except:
                pass

    typestore = get_typestore(Stores.LATEST)
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        # Find image topics
        image_topics = {}
        for connection in reader.connections:
            if connection.msgtype == 'sensor_msgs/msg/Image':
                image_topics[connection.id] = connection

        if not image_topics:
            print(f"  No image topics found in {bag_name}")
            return

        print(f"  Found {len(image_topics)} image topics.")

        # We will write an MP4 for each image topic
        video_writers = {}
        video_dimensions = {}

        for connection, timestamp, rawdata in reader.messages(connections=image_topics.values()):
            topic_name = connection.topic

            # Deserialize the raw data to a ROS message
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Initialize video writer on first frame for each topic
            if topic_name not in video_writers:
                width = msg.width
                height = msg.height
                fps = 30  # Default fps, can be refined based on timestamps if needed

                safe_topic_name = topic_name.strip('/').replace('/', '_')
                out_filename = f"{bag_name}_{safe_topic_name}.mp4"
                out_filepath = os.path.join(out_dir, out_filename)

                # Using imageio to ensure mp4 encoding works on all platforms with bundled ffmpeg
                video_writers[topic_name] = imageio.get_writer(out_filepath, fps=fps, macro_block_size=1)
                print(f"  Initialized video writer for {topic_name}: {out_filepath} (Encoding: {msg.encoding})")
                video_dimensions[topic_name] = (width, height)

            # Convert image data to numpy array
            data = np.frombuffer(msg.data, dtype=np.uint8)

            # Handle different encodings
            try:
                if msg.encoding in ['yuv422', 'yuv422_yuy2']:
                    # YUV422 is 2 bytes per pixel
                    img_np = data.reshape((msg.height, msg.width, 2))
                elif msg.encoding in ['nv12']:
                    # NV12 is 1.5 bytes per pixel, handled by cvtColor if we pass it as flat height*1.5
                    img_np = data.reshape((int(msg.height * 1.5), msg.width))
                elif msg.encoding in ['mono8', '8UC1']:
                    img_np = data.reshape((msg.height, msg.width))
                else:
                    # Generic RGB/BGR: 3 bytes per pixel
                    expected_bpp = len(msg.data) // (msg.width * msg.height)
                    img_np = data.reshape((msg.height, msg.width, expected_bpp))

                # Convert color space if needed (imageio expects RGB)
                conv_code = encoding_to_cvtype(msg.encoding)
                if conv_code is not None:
                    frame = cv2.cvtColor(img_np, conv_code)
                else:
                    if len(img_np.shape) == 2:
                        frame = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    else:
                        frame = img_np

                # Ensure the frame is exactly the size the video writer expects
                if (frame.shape[1], frame.shape[0]) != video_dimensions[topic_name]:
                    frame = cv2.resize(frame, video_dimensions[topic_name])

                video_writers[topic_name].append_data(frame)

            except Exception as e:
                print(f"  Error processing frame on topic {topic_name}: {e}")

        # Release video writers
        for topic_name, writer in video_writers.items():
            writer.close()
            print(f"  Finished writing {topic_name} for bag {bag_name}")


def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bags to MP4 files.')
    parser.add_argument('target_dir', help='Directory containing ROS2 bag subdirectories')
    parser.add_argument('--out_dir', help='Output directory for MP4 files (defaults to target_dir)')
    args = parser.parse_args()

    target_dir = args.target_dir
    out_dir = args.out_dir if args.out_dir else target_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Check if the target dir itself is a bag
    if os.path.exists(os.path.join(target_dir, 'metadata.yaml')):
        process_bag(target_dir, out_dir)
    else:
        # Iterate over all subdirectories
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                # A valid ROS2 bag should have a metadata.yaml or .db3 file inside
                has_metadata = os.path.exists(os.path.join(item_path, 'metadata.yaml'))
                has_db3 = len(glob.glob(os.path.join(item_path, '*.db3'))) > 0

                if has_metadata or has_db3:
                    try:
                        process_bag(item_path, out_dir)
                    except Exception as e:
                        print(f"Failed to process bag {item_path}: {e}")


if __name__ == '__main__':
    main()
