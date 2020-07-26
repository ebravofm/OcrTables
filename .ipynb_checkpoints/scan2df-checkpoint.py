import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

class scan2df:

    def __init__(self, image_path):
        self.original = cv2.imread(image_path)
        self.image = self.original.copy()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.threshold(self.gray, 130, 255, cv2.THRESH_BINARY)[1]
        self.edges = cv2.Canny(self.gray, 50, 70)

        self.shape = self.original.shape
        self.height = self.shape[0]
        self.width = self.shape[1]

        self.hlines = [1, self.height]
        self.vlines = [1, self.width]

        self.houglines = None
        self.drawn = self.original.copy()

        for y in self.hlines:
                cv2.line(self.drawn,(1,y),(self.width-1,y),(255,0,0),2)
        for x in self.vlines:
                cv2.line(self.drawn,(x,1),(x,self.height-1),(255,0,0),2)


    def add_vline(self, x_list):
        if  isinstance(x_list,int):
            x_list = [x_list]
        for x in x_list:
            cv2.line(self.image,(x,0),(x,2340), (255,0,0),2)
            self.vlines.append(x)
            self.vlines.sort()


    def add_hline(self, y_list):
        if  isinstance(y_list,int):
            y_list = [y_list]
        for y in y_list:
            cv2.line(self.image,(0,y),(4160,y), (255,0,0),2)
            self.hlines.append(y)
            self.hlines.sort()


    def remove_vline(self, a):
        self.vlines = [x for x in self.vlines if x!=a]


    def remove_hline(self, a):
        self.hlines = [x for x in self.hlines if x!=a]


    def remove_duplicates(self, lista, distance=25):
        d = lista.copy()
        d.sort()
        diff = [y - x for x, y in zip(*[iter(d)] * 2)]
        m = [[d[0]]]

        for x in d[1:]:
            if x - m[-1][0] < distance:
                m[-1].append(x)
            else:
                m.append([x])
        n = [int(np.mean(l)) for l in m]
        return n


    def reset_gray(self):
        img = self.original.copy()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def reset_thresh(self, th=130):
        img = self.gray.copy()        
        self.thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]


    def reset_edge(self, th_min=50, th_max=0, dif=20):
        img = self.gray.copy()        
        if th_max == 0:
            th_max = th_min + dif
        self.edges = cv2.Canny(img, th_min, th_max)


    def reset_drawn(self):
        self.hlines = [1, self.height]
        self.vlines = [1, self.width]
        self.drawn = self.original.copy()

        for y in self.hlines:
                cv2.line(self.drawn,(1,y),(self.width-1,y),(255,0,0),2)
        for x in self.vlines:
                cv2.line(self.drawn,(x,1),(x,self.height-1),(255,0,0),2)


    def lines(self,
              threshold = 100,
              minLineLength = 20,
              maxLineGap = 0,
              distance = 25):

        self.houghlines = cv2.HoughLinesP(self.edges,
                                          rho = 1,
                                          theta = 1 * np.pi / 180,
                                          threshold = threshold,
                                          minLineLength = minLineLength,
                                          maxLineGap = maxLineGap)

        for line in self.houghlines:
            if line[0][0] == line[0][2]:
                self.vlines.append(line[0][0])

            if line[0][1] == line[0][3]:
                self.hlines.append(line[0][1])

        self.hlines = self.remove_duplicates(self.hlines)
        self.vlines = self.remove_duplicates(self.vlines)


    def apply_lines(self):

        for y in self.hlines:
                cv2.line(self.drawn,(1,y),(self.width-2,y),(255,0,0),2)
        for x in self.vlines:
                cv2.line(self.drawn,(x,1),(x,self.height-2),(255,0,0),2)


    def ocr(self):
        
        self.lines()
        
        self.apply_lines()
        
        thresh = self.thresh.copy()

        for y in self.hlines:
                cv2.line(thresh,(1,y),(self.width-2,y),(255,0,0),2)
        for x in self.vlines:
                cv2.line(thresh,(x,1),(x,self.height-2),(255,0,0),2)

        df = pd.DataFrame(np.zeros((len(self.hlines)-1, len(self.vlines)-1)))
        for i in range(len(self.hlines)):
            print('row {}...'.format(i))
            for j in range(len(self.vlines)):
                try:
                    t = pytesseract.image_to_string(thresh[self.hlines[i]:self.hlines[i+1],self.vlines[j]:self.vlines[j+1]], lang='spa')
                    df.iloc[i,j] = t
                except Exception as exc:
                    print(exc)
        return df


cv2.imwrite('thresh.png', thresh)
