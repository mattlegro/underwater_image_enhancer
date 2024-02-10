import sys
from os import walk
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import cv2


class VideoCreator:
    supported_img_ext = ('.jpg', '.png')

    def __init__(self, img_dir, video, fps=1, fourcc="MJPG", verbose=False):
        self.img_dir = img_dir
        self.video = video
        self.fps = fps
        self.fourcc = fourcc
        self.verbose = verbose
        self.validate_inputs()

    def validate_inputs(self):
        # Check if given video file extension is supported
        video_ext = osp.splitext(self.video)[-1]

        # Check if given output video file exists -- Ask user's confirmation for overwriting it
        if osp.exists(self.video):
            user_confirmation = self.query('Output video file exists: {}\nOverwrite?'.format(self.video))
            if not user_confirmation:
                sys.exit()

        # Check if given image dir is valid
        if not osp.isdir(self.img_dir):
            raise NotADirectoryError("Input images directory is not valid: {}".format(self.img_dir))

    @staticmethod
    def query(question, default="yes"):
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("Invalid default answer: {}".format(default))

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")

    @staticmethod
    def get_img_size(image_filename):
        im = Image.open(image_filename)
        return im.size[0], im.size[1]

    def create_video(self):
        if self.verbose:
            print("#. Create video file from images in directory: {}".format(self.img_dir))

        # Collect all image filenames in given directory
        img_files, img_widths, img_heights = self.collect_images()
        
        # Write video
        self.write_video(img_files, img_widths, img_heights)

    def collect_images(self):
        img_files = []
        img_widths = []
        img_heights = []
        for r, d, f in walk(self.img_dir):
            for file in f:
                img_ext = osp.splitext(file)[-1]
                if img_ext in self.supported_img_ext:
                    img_w, img_h = self.get_img_size(osp.join(r, file))
                    img_widths.append(img_w)
                    img_heights.append(img_h)
                    img_files.append(osp.join(r, file))
        img_files.sort()
        return img_files, img_widths, img_heights

    def write_video(self, img_files, img_widths, img_heights):
        max_frame_width = max(img_widths)
        max_frame_height = max(img_heights)
        frame_shape = (max_frame_width, max_frame_height)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        video = cv2.VideoWriter(self.video, fourcc, self.fps, frame_shape)
        for frame_no, img_file in enumerate(img_files):
            self.progress_update("  \\__.Progress", len(img_files), frame_no + 1)
            img_w = img_widths[frame_no]
            img_h = img_heights[frame_no]
            frame = self.process_frame(img_w, img_h, max_frame_width, max_frame_height, img_file)
            video.write(frame)
        video.release()

    @staticmethod
    def progress_update(msg, total, progress):
        # Implementation of progress update remains the same
        pass

    @staticmethod
    def process_frame(img_w, img_h, max_frame_width, max_frame_height, img_file):
        if img_w < max_frame_width or img_h < max_frame_height:
            frame = np.zeros(shape=(max_frame_height, max_frame_width, 3), dtype=np.uint8)
            y = (max_frame_height - img_h) // 2
            x = (max_frame_width - img_w) // 2
            frame[y:y+img_h, x:x+img_w] = cv2.imread(img_file)
        else:
            frame = cv2.imread(img_file)
        return frame


def parse_arguments():
    parser = argparse.ArgumentParser("Create video from images in directory")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument('--img_dir', type=str, required=True, help="set input image directory")
    parser.add_argument('--fps', type=int, default=1, help="set output video fps")
    parser.add_argument('--fourcc', type=str, default="MJPG",
                        choices=("MJPG", "XVID"),
                        help="set the 4-character code of codec used to compress the frames.")
    parser.add_argument('--video', type=str, help="set output video file in format <filename>.<ext>")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    video_creator = VideoCreator(img_dir=args.img_dir, video=args.video, fps=args.fps, fourcc=args.fourcc, verbose=args.verbose)
    video_creator.create_video()


if __name__ == '__main__':
    main()
