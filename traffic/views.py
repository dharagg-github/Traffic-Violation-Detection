from django.shortcuts import render
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import traffic.object_detection as od
import imageio
import cv2
import os.path as osp
import glob
import torch
import ESRGAN.RRDBNet_arch as arch
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


def index(request):
    return render(request, '../templates/index.html')

def detect(request):
    class Window(Frame):

        def __init__(self, master=None):
            Frame.__init__(self, master)

            self.master = master
            self.pos = []
            self.line = []
            self.rect = []
            self.master.title("GUI")
            self.pack(fill=BOTH, expand=1)

            self.counter = 0

            menu = Menu(self.master)
            self.master.config(menu=menu)

            file = Menu(menu)
            file.add_command(label="Open", command=self.open_file)
            file.add_command(label="Exit", command=self.client_exit)
            menu.add_cascade(label="File", menu=file)

            analyze = Menu(menu)
            analyze.add_command(label="Region of Interest", command=self.regionOfInterest)
            menu.add_cascade(label="Analyze", menu=analyze)

            self.filename = "C:\\Users\\DHARANEESH GG\\Downloads\\Traffic-Signal-Violation-Detection-System-master\\Images\\home.jpg"
            self.imgSize = Image.open(self.filename)
            self.tkimage = ImageTk.PhotoImage(self.imgSize)
            self.w, self.h = (1366, 768)

            self.canvas = Canvas(master=root, width=self.w, height=self.h)
            self.canvas.create_image(20, 20, image=self.tkimage, anchor='nw')
            self.canvas.pack()

        def open_file(self):
            self.filename = filedialog.askopenfilename()

            cap = cv2.VideoCapture(self.filename)

            reader = imageio.get_reader(self.filename)
            fps = reader.get_meta_data()['fps']

            ret, image = cap.read()

            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                count += 1
                if count % 3 != 0:
                    continue

            cv2.imwrite('C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/Images/preview.jpg',
                        image)

            self.show_image('C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/Images/preview.jpg')

        def show_image(self, frame):
            self.imgSize = Image.open(frame)
            self.tkimage = ImageTk.PhotoImage(self.imgSize)
            self.w, self.h = (1366, 768)

            self.canvas.destroy()

            self.canvas = Canvas(master=root, width=self.w, height=self.h)
            self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
            self.canvas.pack()

        def regionOfInterest(self):
            root.config(cursor="plus")
            self.canvas.bind("<Button-1>", self.imgClick)

        def client_exit(self):
            exit()

        def imgClick(self, event):

            if self.counter < 2:
                x = int(self.canvas.canvasx(event.x))
                y = int(self.canvas.canvasy(event.y))
                self.line.append((x, y))
                self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
                self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
                self.counter += 1

            # elif self.counter < 4:
            #     x = int(self.canvas.canvasx(event.x))
            #     y = int(self.canvas.canvasy(event.y))
            #     self.rect.append((x, y))
            #     self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            #     self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            #     self.counter += 1

            if self.counter == 2:
                # unbinding action with mouse-click
                self.canvas.unbind("<Button-1>")
                root.config(cursor="arrow")
                self.counter = 0

                # show created virtual line
                print(self.line)
                print(self.rect)
                img = cv2.imread(
                    'C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/Images/preview.jpg')
                cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
                cv2.imwrite('C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/Images/copy.jpg', img)
                self.show_image('C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/Images/copy.jpg')

                ## for demonstration
                # (rxmin, rymin) = self.rect[0]
                # (rxmax, rymax) = self.rect[1]

                # tf = False
                # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymin), (rxmin, rymax))
                # print(tf)
                # tf |= self.intersection(self.line[0], self.line[1], (rxmax, rymin), (rxmax, rymax))
                # print(tf)
                # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymin), (rxmax, rymin))
                # print(tf)
                # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymax), (rxmax, rymax))
                # print(tf)

                # cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)

                # if tf:
                #     cv2.rectangle(img, (rxmin,rymin), (rxmax,rymax), (255,0,0), 3)
                # else:
                #     cv2.rectangle(img, (rxmin,rymin), (rxmax,rymax), (0,255,0), 3)

                # cv2.imshow('traffic violation', img)

                # image processing
                self.main_process()
                print("Executed Successfully!!!")

                # clearing things
                self.line.clear()
                self.rect.clear()
                for i in self.pos:
                    self.canvas.delete(i)

        def intersection(self, p, q, r, t):
            print(p, q, r, t)
            (x1, y1) = p
            (x2, y2) = q

            (x3, y3) = r
            (x4, y4) = t

            a1 = y1 - y2
            b1 = x2 - x1
            c1 = x1 * y2 - x2 * y1

            a2 = y3 - y4
            b2 = x4 - x3
            c2 = x3 * y4 - x4 * y3

            if (a1 * b2 - a2 * b1 == 0):
                return False
            print((a1, b1, c1), (a2, b2, c2))
            x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
            y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
            print((x, y))

            if x1 > x2:
                tmp = x1
                x1 = x2
                x2 = tmp
            if y1 > y2:
                tmp = y1
                y1 = y2
                y2 = tmp
            if x3 > x4:
                tmp = x3
                x3 = x4
                x4 = tmp
            if y3 > y4:
                tmp = y3
                y3 = y4
                y4 = tmp

            if x >= x1 and x <= x2 and y >= y1 and y <= y2 and x >= x3 and x <= x4 and y >= y3 and y <= y4:
                return True
            else:
                return False

        def main_process(self):

            video_src = self.filename

            cap = cv2.VideoCapture(video_src)

            reader = imageio.get_reader(video_src)
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(
                'C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/Output/output.mp4',
                fps=fps)

            j = 1
            while True:
                ret, image = cap.read()

                if (type(image) == type(None)):
                    writer.close()
                    break

                image_h, image_w, _ = image.shape
                new_image = od.preprocess_input(image, od.net_h, od.net_w)

                # run the prediction
                yolos = od.yolov3.predict(new_image)
                boxes = []

                for i in range(len(yolos)):
                    # decode the output of the network
                    boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h,
                                              od.net_w)

                # correct the sizes of the bounding boxes
                od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)

                # suppress non-maximal boxes
                od.do_nms(boxes, od.nms_thresh)

                # draw bounding boxes on the image using labels
                image2 = od.draw_boxes(image, boxes, self.line, od.labels, od.obj_thresh, j)

                writer.append_data(image2)

                # cv2.imwrite('E:/Virtual Traffic Light Violation Detection System/Images/frame'+str(j)+'.jpg', image2)
                # self.show_image('E:/Virtual Traffic Light Violation Detection System/Images/frame'+str(j)+'.jpg')

                cv2.imshow('Traffic Violation', image2)

                print(j)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    writer.close()
                    break

                j = j + 1

            cv2.destroyAllWindows()

    root = Tk()
    app = Window(root)
    root.geometry("%dx%d" % (535, 380))
    root.title("Traffic Violation")

    root.mainloop()


def einfo(request):

    theft_noid = ['jar606l','arc234l']

    img = cv2.imread('C:\\Users\\DHARANEESH GG\\PycharmProjects\\Traffic_Recog\\Media\\Detected_Images\\violation_84.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    cv2.imwrite('C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/ESRGAN/LR/numberplate.jpg',cropped_image)
    # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    model_path = 'C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    # device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    device = torch.device('cpu')

    test_img_folder = 'C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/ESRGAN/LR/*'

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/ESRGAN/results/{:s}_rlt.jpg'.format(base), output)

        reader = easyocr.Reader(['en'])
        img_read = cv2.imread("C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/ESRGAN/results/numberplate_rlt.jpg")
        result = reader.readtext(img_read)
        #print(result)
        strng = result[0][-2]
        print(strng)

        if strng in theft_noid:

            text_file = open("C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/theft_vehicle.txt", "w")
            text_file.write(strng)
            text_file.close()

        else:
            text_file = open("C:/Users/DHARANEESH GG/PycharmProjects/Traffic_Recog/Media/violated_vehicle.txt", "w")
            text_file.write(strng)
            text_file.close()
    return render(request, '../templates/success.html')

