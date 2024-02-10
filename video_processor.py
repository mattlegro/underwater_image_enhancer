import sys
import cv2
import argparse
import os

class VideoProcessor:
    def __init__(self, path, threshold=100.0, step=1, save=""):
        self.path = path
        self.threshold = threshold
        self.step = step
        self.save = save
        self.count = 0
        self.blurryFrame = 0
        self.savedFrame = 0
        self.frameStep = 0

    @staticmethod
    def variance_of_laplacian(image):
        """
        compute the Laplacian of the image and return the focus measure
        """
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def process_video(self):
        if not self.path:
            sys.exit("Please supply a video file '-p <path>'")

        vidcap = cv2.VideoCapture(self.path)

        success, image = vidcap.read()

        print("Working...")

        while success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = self.variance_of_laplacian(gray)

            if self.frameStep == self.step:
                if fm > self.threshold:
                    self.savedFrame += 1
                    image_name = "{:05d}.png".format(self.count)
                    image_path = os.path.join(self.save, image_name)
                    cv2.imwrite(image_path, image)
                self.frameStep = 0

            if fm < self.threshold:
                self.blurryFrame += 1

            success, image = vidcap.read()
            self.count += 1
            self.frameStep += 1
        print("DONE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-p", "--path", type=str, default="",
                          help="path to the video file", required=True)
    optional.add_argument("-t", "--threshold", type=float, default=100.0,
                          help="default threshold is 100.0. Use 10-30 for motion")
    optional.add_argument("-s", "--step", type=int,
                          default=1, help="frame step size")
    optional.add_argument("--save", default= "", type= str, help= "path to save the frames in a directory")
    args = vars(parser.parse_args())

    processor = VideoProcessor(args["path"], args["threshold"], args["step"], args["save"])
    processor.process_video()
