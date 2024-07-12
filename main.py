import cv2
import numpy as np
import argparse
from scipy.special import comb

# args setting
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--input", help="input file video")
parser.add_argument('--leftPoint', type=int, help="Left rail offset", default=450)
parser.add_argument('--rightPoint', type=int, help="Right rail offset", default=840)
parser.add_argument('--topPoint', type=int, help="Top rail offset", default=330)
args = parser.parse_args()


def main():
   
    cap = VideoCapture(args.input)

    # initialization for line detection
    expt_startLeft = args.leftPoint
    expt_startRight = args.rightPoint
    expt_startTop = args.topPoint

    # value initialize
    left_maxpoint = [0] * 50
    right_maxpoint = [195] * 50

    # convolution filter
    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    # Next frame availability
    r = True
    first = True

    while r is True:
        r, frame = cap.read()
        if frame is None:
            break

        # cut away invalid frame area
        valid_frame = frame[expt_startTop:, expt_startLeft:expt_startRight]
        # original_frame = valid_frame.copy()

        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)

        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # apply gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        # merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # initialization for sliding window property
        sliding_window = [20, 190, 200, 370]
        slide_interval = 15
        slide_height = 15
        slide_width = 60

        # initialization for bezier curve variables
        left_points = []
        right_points = []

      
        count = 0
        for i in range(340, 40, -slide_interval):
            # edges in sliding window
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

            # left railroad line processing
            if left_edge.argmax() > 0:
                left_maxindex = sliding_window[0] + left_edge.argmax()
                left_maxpoint[count] = left_maxindex
                cv2.line(valid_frame, (left_maxindex, i + int(slide_height / 2)),
                         (left_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                left_points.append([left_maxindex, i + int(slide_height / 2)])
                sliding_window[0] = max(0, left_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[1] = min(390, left_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                cv2.rectangle(valid_frame, (sliding_window[0], i + slide_height), (sliding_window[1], i), (0, 255, 0),
                              1)

            # right railroad line processing
            if right_edge.argmax() > 0:
                right_maxindex = sliding_window[2] + right_edge.argmax()
                right_maxpoint[count] = right_maxindex
                cv2.line(valid_frame, (right_maxindex, i + int(slide_height / 2)),
                         (right_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                right_points.append([right_maxindex, i + int(slide_height / 2)])
                sliding_window[2] = max(0, right_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[3] = min(390, right_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                cv2.rectangle(valid_frame, (sliding_window[2], i + slide_height), (sliding_window[3], i), (0, 0, 255),
                              1)
            count += 1

        # bezier curve process
        bezier_left_xval, bezier_left_yval = bezier_curve(left_points, 50)
        bezier_right_xval, bezier_right_yval = bezier_curve(right_points, 50)

        bezier_left_points = []
        bezier_right_points = []
        try:
            old_point = (bezier_left_xval[0], bezier_left_yval[0])
            for point in zip(bezier_left_xval, bezier_left_yval):
                cv2.line(valid_frame, old_point, point, (0, 0, 255), 2, cv2.LINE_AA)
                old_point = point
                bezier_left_points.append(point)

            old_point = (bezier_right_xval[0], bezier_right_yval[0])
            for point in zip(bezier_right_xval, bezier_right_yval):
                cv2.line(valid_frame, old_point, point, (255, 0, 0), 2, cv2.LINE_AA)
                old_point = point
                bezier_right_points.append(point)
        except IndexError:
            pass
       
        cv2.imshow('Video', valid_frame)
        cv2.waitKey(1)
    print('finish')


#for reading video
class VideoCapture:
    def __init__(self, path):
        
        path= 'video/test2.mp4'
        self.video = cv2.VideoCapture(path)
       

    def __del__(self):
        self.video.release()

    def read(self):
        # single frame of video
        ret, frame = self.video.read()
        return frame is not None, frame


# bezier curve function
def bezier_curve(points, ntimes=1000):

    def bernstein_poly(i, n, t):
        
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, ntimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals.astype('int32'), yvals.astype('int32')


def nothing(value):
    pass


if __name__ == '__main__':
    main()
