import cv2
import os
import argparse
from tqdm import tqdm
from glob import glob


def make_parser():
    parser = argparse.ArgumentParser("Extract Image From Video")

    parser.add_argument("--video_path", type=str, required=True, help="the path of the video")
    parser.add_argument("--image_path", type=str, required=True, help="the image path to save")
    parser.add_argument("--folder", action="store_true",
                        help="the mode to extract image from the folder contains the video")

    return parser


def get_frame(video_path, _image_folder):
    cam = cv2.VideoCapture(video_path)
    current_frame = 0
    while True:
        # reading from frame
        ret, frame = cam.read()
        if ret:
            save_name = f"{image_folder}/{str(current_frame).zfill(4)}.png"
            cv2.imwrite(save_name, frame)

            current_frame += 1
        else:
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = make_parser().parse_args()

    if not os.path.exists(args.image_path):
        os.mkdir(args.image_path)

    if args.folder:
        for video in tqdm(glob(f"{args.video_path}/*/*")):
            cls_name = video.split("/")[-2]

            image_folder = f"{args.image_path}/{cls_name}"
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)

            get_frame(video, image_folder)
    else:
        get_frame(args.video_path, args.image_path)
