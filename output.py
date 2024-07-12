import cv2
import numpy as np
import argparse
from scipy.special import comb
import os

def main(args):
    cap = VideoCapture(args.input)

    expt_start_left = args.leftPoint
    expt_start_right = args.rightPoint
    expt_start_top = args.topPoint

    left_maxpoint = [0] * 50
    right_maxpoint = [195] * 50

    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    r = True
    first = True

    # Define output video parameters
    output_filename = 'output_video1.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Adjust as needed
    frame_width = 640  # Default width
    frame_height = 480  # Default height

    # Create VideoWriter object
    out = None

    while r is True:
        r, frame = cap.read()
        if frame is None:
            break

        valid_frame = frame[expt_start_top:, expt_start_left:expt_start_right]

        if out is None:
            frame_height, frame_width, _ = valid_frame.shape
            out = cv2.VideoWriter(output_filename, codec, fps, (frame_width, frame_height))

        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)
        histeqaul_frame = cv2.equalizeHist(gray_frame)
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        sliding_window = [20, 190, 200, 370]
        slide_interval = 15
        slide_height = 15
        slide_width = 60

        left_points = []
        right_points = []

        count = 0
        for i in range(340, 40, -slide_interval):
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

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

        # Bezier curve process
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

        # Write the frame to the output video
        out.write(valid_frame)

        cv2.imshow('Video', valid_frame)
        cv2.waitKey(1)

    # Release the VideoWriter object
    out.release()
    print('Finish')

    # Move the output video to the desired location
    os.replace(output_filename, os.path.join(args.output_dir, output_filename))
    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")


def detect_obstacles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 100:  # Adjust threshold as needed
            obstacles.append(((x, y), (x + w, y + h)))

    return obstacles


class VideoCapture:
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def read(self):
        ret, frame = self.video.read()
        return frame is not None, frame


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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', "--input", help="input file video")
    parser.add_argument('-o', "--output_dir", help="output directory to save the video", default='.')
    parser.add_argument('--leftPoint', type=int, help="Left rail offset", default=450)
    parser.add_argument('--rightPoint', type=int, help="Right rail offset", default=840)
    parser.add_argument('--topPoint', type=int, help="Top rail offset", default=330)
    args = parser.parse_args()

    main(args)
