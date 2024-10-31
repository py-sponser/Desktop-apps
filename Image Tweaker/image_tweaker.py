import cv2, threading
import numpy as np
import matplotlib.pyplot as plt


class ImageTweaker:
    def __init__(self):
        """Initially defining required attributes to handle processing the image"""
        self.image_path = str()
        self.image = str()
        self.image_states = list()
        self.planes = str()
        self.blur_filters = {"gauss": self.gauss_filter, "average": self.average_filter,
                             "circular": self.circular_filter, "pyramidal": self.pyramidal_filter,
                             "cone": self.cone_filter, "median": self.median_blur}
        self.filter_points = None
        self.range_dst = float()
        self.filtered_image = str()

    def set_image_conf(self, image_path):
        """Setting configurations of Image Tweaker to handle processing the image"""
        self.image = cv2.imread(image_path)  # reading image by its path
        self.image_states.append(self.image)  # saving the initial image state
        self.image_path = image_path  # saving the image path

    def switch_to_grayed_scale(self):
        """Converting the image to gray scale"""
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_states.append(self.image)
        return self.image

    def get_image_data(self):
        """Getting the image properties"""
        if len(self.image.shape) < 3:
            min_pixel_amplitude = min([min(pixelComp) for pixelComp in self.image])
            max_pixel_amplitude = max([max(pixelComp) for pixelComp in self.image])
            average_pixels = round(sum([sum(pixelComp) for pixelComp in self.image]) / self.image.size, 2)
        else:
            min_pixel_amplitude = min([min(comp) for pixelComp in self.image for comp in pixelComp])
            max_pixel_amplitude = max([max(comp) for pixelComp in self.image for comp in pixelComp])
            average_pixels = round(sum([sum(comp) for pixelComp in self.image for comp in pixelComp]) / self.image.size, 2)

        img_data = [
            # returns the total pixels (rows * cols)
            f"[+] Size of Grayed Image: {self.image.size}",
            f"[+] Grayed Image Rows: {self.image.shape[0]}",
            f"[+] Grayed Image Cols: {self.image.shape[1]}",
            f"[+] Datatype of Grayed Image: {self.image.dtype}",
            f"[+] Amplitude at x=0, y=0: {self.image[0, 0]}",
            f"[+] Minimum Pixel Amplitude: {min_pixel_amplitude}",
            f"[+] Maximum Pixel Amplitude: {max_pixel_amplitude}",
            f"[+] Image Average: {average_pixels}",
        ]
        return img_data

    def translate_image(self, tx, ty):
        """Translating the image with a given tx, ty"""
        height, width = self.image.shape[:2]
        translation_matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty],
        ])
        self.image = cv2.warpAffine(self.image, translation_matrix, (width, height))
        self.image_states.append(self.image)
        return self.image

    def rotate_image(self, angle):
        """Rotating Image with a given angel"""
        height, width = self.image.shape[:2]
        center = (width/2, height/2)
        rotating_matrix = cv2.getRotationMatrix2D(center=center, angle=int(angle), scale=1) # scale, resize window as image size
        self.image = cv2.warpAffine(src=self.image, M=rotating_matrix, dsize=(width, height))
        self.image_states.append(self.image)
        return self.image

    def skew_image(self, skew_value):
        """Skewing the image"""
        height, width = self.image.shape[:2]
        src_points = np.float32([(0,0), (width, 0), (-float(skew_value), height)]) # 3 points, (x=0,y=0), (x=width, y=0), (x=-100, y=height)
        dst_points = np.float32([(0,0), (width, 0), (0, height)])
        skewing_matrix = cv2.getAffineTransform(src_points, dst_points)
        self.image = cv2.warpAffine(self.image, skewing_matrix, (width, height))
        self.image_states.append(self.image)
        return self.image

    def deskew_image(self, skew_value):
        height, width = self.image.shape[:2]
        src_points = np.float32([(0,0), (width, 0), (0, height)])
        dst_points = np.float32([(0,0), (width, 0), (-float(skew_value), height)])
        skewing_matrix = cv2.getAffineTransform(src_points, dst_points)
        self.image = cv2.warpAffine(self.image, skewing_matrix, (width, height))
        self.image_states.append(self.image)
        return self.image

    def zoom_click_event(self, event, x, y, flags, zoom_range):
        print("Hello World")
        if event == cv2.EVENT_LBUTTONDOWN:
            r = int(zoom_range)
            cropped_image = self.image[x-r:x+r, y-r:y+r]
            self.image = cv2.resize(cropped_image, (self.image.shape[1],self.image.shape[0]), fx=2, fy=2)
            cv2.imshow("Zoomed Image", self.image)
            cv2.imwrite(self.image_path, self.image)
            self.image_states.append(self.image)

    def zoom_image(self, zoom_range):
        """Zooming the image"""
        print("Hello")
        if zoom_range:
            cv2.imshow("image", self.image)
            cv2.setMouseCallback("image", self.zoom_click_event, param=zoom_range)
            cv2.waitKey()
            cv2.destroyAllWindows()



    def flip_image(self, axis_code):
        """Flipping the image with a given axis code, 0, 1, or -1"""
        self.image = cv2.flip(self.image, axis_code)
        self.image_states.append(self.image)
        return self.image

    def blind_image_points(self,second_image):
        height1, width1 = self.image.shape[:2]
        dst = np.ndarray((height1, width1), np.uint8)
        second_image = cv2.resize(second_image, (width1, height1))

        for i in range(0, height1):
            for j in range(0, width1):
                dst[i][j] = (self.image[i][j]*0.6) + second_image[i][j]*0.4
        self.image = dst
        self.image_states.append(self.image)
        return self.image

    def histogram_display(self):
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        plt.hist(hist, 256, [0, 256])
        plt.savefig(f"web/images/histogram.png")

    def histogram_equalize(self):
        self.image = cv2.imread(self.image_path, 0)
        self.image = cv2.equalizeHist(self.image)
        self.image_states.append(self.image)
        return self.image

    def correct_gama_power(self, power):
        """Brightness over high range"""
        self.image = np.array(255*(self.image/255)**float(power), dtype="uint8")
        self.image_states.append(self.image)
        return self.image

    def transform_linear(self):
        """Brightness over small range"""
        self.image = 255 - self.image
        self.image_states.append(self.image)
        return self.image

    def log_transform(self):
        """Brightness over medium range"""
        c = 255 / np.log(1 + np.max(self.image))
        self.image = c * np.log(self.image + 1)
        self.image = np.array(self.image, dtype="uint8")
        self.image_states.append(self.image)
        return self.image

    def slice_gray_level(self):
        """Clarify the image of gray scale"""
        height, width = self.image.shape[:2]
        temp_image = np.zeros((height, width), dtype="uint8")
        min_range = 50
        max_range = 100
        for i in range(height):
            for j in range(width):
                if self.image[i, j] > max_range:
                    temp_image[i, j] = 255
                elif self.image[i, j] < min_range:
                    temp_image[i, j] = 0

        self.image_states.append(self.image)
        return temp_image

    def compress(self):
        temp_8bit_amplitudes = [np.binary_repr(self.image[i, j], width=8) for i in range(self.image.shape[0]) for j in range(self.image.shape[1])]
        eight_bits_image = (np.array([int(i[0]) for i in temp_8bit_amplitudes], dtype="uint8") * 128).reshape(self.image.shape[0], self.image.shape[1])
        seven_bits_image = (np.array([int(i[1]) for i in temp_8bit_amplitudes], dtype="uint8") * 64).reshape(self.image.shape[0], self.image.shape[1])
        six_bits_image = (np.array([int(i[2]) for i in temp_8bit_amplitudes], dtype="uint8") * 32).reshape(self.image.shape[0], self.image.shape[1])
        five_bits_image = (np.array([int(i[3]) for i in temp_8bit_amplitudes], dtype="uint8") * 16).reshape(self.image.shape[0], self.image.shape[1])
        four_bits_image = (np.array([int(i[4]) for i in temp_8bit_amplitudes], dtype="uint8") * 8).reshape(self.image.shape[0], self.image.shape[1])
        three_bits_image = (np.array([int(i[5]) for i in temp_8bit_amplitudes], dtype="uint8") * 4).reshape(self.image.shape[0], self.image.shape[1])
        two_bits_image = (np.array([int(i[6]) for i in temp_8bit_amplitudes], dtype="uint8") * 2).reshape(self.image.shape[0], self.image.shape[1])
        one_bit_image = (np.array([int(i[7]) for i in temp_8bit_amplitudes], dtype="uint8") * 1).reshape(self.image.shape[0], self.image.shape[1])
        eight_planes = [eight_bits_image, seven_bits_image, six_bits_image, five_bits_image, four_bits_image, three_bits_image, two_bits_image,
                        one_bit_image]
        # returning 8bit image
        # #Concatenate these images for ease of display using cv2.hconcat()
        finalr = cv2.hconcat([eight_bits_image, seven_bits_image, six_bits_image, five_bits_image])
        finalv = cv2.hconcat([four_bits_image, three_bits_image, two_bits_image, one_bit_image])

        # Vertically concatenate
        self.image = cv2.vconcat([finalr,finalv])
        self.planes = self.image
        self.image = eight_bits_image + seven_bits_image + six_bits_image + five_bits_image
        self.image_states.append(self.image)
        return self.image

    def show_compress_planes(self):
        return self.planes

    def filter_click_event(self, event, x, y, flags, filter_type):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, " ", y)
            self.filter_points = {"x": x, "y": y}
            self.filtered_image = self.blur_filters[filter_type]()
            cv2.imshow("Filter Blurred Image", self.filtered_image)
            cv2.imwrite(self.image_path, self.filtered_image)

    def filter(self, filter_type, pixel_range_dst):
        if not pixel_range_dst:
            filtered_image = self.blur_filters[filter_type]()
            self.filter_points = None
            return filtered_image
        else:
            self.range_dst = int(pixel_range_dst)
            cv2.imshow("image", self.image)
            cv2.setMouseCallback("image", self.filter_click_event, param=filter_type)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.image_states.append(self.image)

    def gauss_filter(self):
        gauss_kernel = np.array([[1,2,1], [2,4,2], [1,2,1]], np.float32) / 16
        if self.filter_points is not None:
            x = self.filter_points["x"]
            y = self.filter_points["y"]
            dst = self.range_dst
            print("Hello")
            self.image[y:y+dst, x:x+dst] = cv2.filter2D(self.image[y:y+dst, x:x+dst], cv2.CV_8UC1, gauss_kernel)
        else:
            self.image = cv2.filter2D(self.image, cv2.CV_8UC1, gauss_kernel)

        return self.image

    def average_filter(self):
        averaging_kernel = np.array([[1,1,1], [1,1,1], [1,1,1]], np.float32) / 9

        if self.filter_points is not None:
            x = self.filter_points["x"]
            y = self.filter_points["y"]
            dst = self.range_dst
            self.image[y:y+dst, x:x+dst] = cv2.filter2D(self.image[y:y+dst, x:x+dst], cv2.CV_8UC1, averaging_kernel)
        else:
            self.image = cv2.filter2D(self.image, cv2.CV_8UC1, averaging_kernel)

        return self.image

    def circular_filter(self):
        circular_kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]
                                   , np.float32) / 21
        if self.filter_points is not None:
            x = self.filter_points["x"]
            y = self.filter_points["y"]
            dst = self.range_dst
            self.image[y:y+dst, x:x+dst] = cv2.filter2D(self.image[y:y+dst, x:x+dst], cv2.CV_8UC1, circular_kernel)
        else:
            self.image = cv2.filter2D(self.image, cv2.CV_8UC1, circular_kernel)

        return self.image

    def pyramidal_filter(self):
        pyramidal_kernel = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3],
                                     [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]], np.float32) / 81

        if self.filter_points is not None:
            x = self.filter_points["x"]
            y = self.filter_points["y"]
            dst = self.range_dst
            self.image[y:y+dst, x:x+dst] = cv2.filter2D(self.image[y:y+dst, x:x+dst], cv2.CV_8UC1, pyramidal_kernel)
        else:
            self.image = cv2.filter2D(self.image, cv2.CV_8UC1, pyramidal_kernel)

        return self.image

    def cone_filter(self):
        cone_kernel = np.array([[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1],
                               [0, 2, 2, 2, 0], [0, 0, 1, 0, 0]], np.float32) / 25

        if self.filter_points is not None:
            x = self.filter_points["x"]
            y = self.filter_points["y"]
            dst = self.range_dst
            self.image[y:y+dst, x:x+dst] = cv2.filter2D(self.image[y:y+dst, x:x+dst], cv2.CV_8UC1, cone_kernel)
        else:
            self.image = cv2.filter2D(self.image, cv2.CV_8UC1, cone_kernel)

        return self.image

    def median_blur(self):
        if self.filter_points is not None:
            x = self.filter_points["x"]
            y = self.filter_points["y"]
            dst = self.range_dst
            self.image[y:y+dst, x:x+dst] = cv2.medianBlur(self.image[y:y+dst, x:x+dst], 5)
        else:
            self.image = cv2.medianBlur(self.image, 5)

        return self.image

    def sobel_filter(self, sobel_filter_type):
        sobel_filter_types = ["horizontal", "vertical", "combined"]
        sobel_filter_type = str(sobel_filter_type).lower()
        if sobel_filter_type in sobel_filter_types:
            vertical_edgies = cv2.Sobel(self.image, cv2.CV_16UC1, 1, 0, 5)
            horizontal_edgies = cv2.Sobel(self.image, cv2.CV_16UC1, 0, 1, 5)
            if sobel_filter_type == "horizontal":
                self.image = cv2.convertScaleAbs(horizontal_edgies)

            elif sobel_filter_type == "vertical":
                self.image = cv2.convertScaleAbs(vertical_edgies)

            elif sobel_filter_type == "combined":
                vertical_edgies = cv2.convertScaleAbs(vertical_edgies)
                horizontal_edgies = cv2.convertScaleAbs(horizontal_edgies)
                self.image = cv2.addWeighted(vertical_edgies, 1, horizontal_edgies, 1, 0.0)

        self.image_states.append(self.image)
        return self.image

    def laplace(self):
        edgies = cv2.Laplacian(self.image, cv2.CV_16UC1, 3)
        self.image = cv2.convertScaleAbs(edgies)
        self.image_states.append(self.image)
        return self.image

    def segment_image(self):
        # thresholding, loop through every amplitude of image, if amp < T, amp = 0, if amp > T, amp = 255, T is determined through track power.
        # edge based, guass filter on the image, then pass it to laplacian
        gauss_kernel = np.array([[1,2,1], [2,4,2], [1,2,1]], np.float32) / 16
        self.image = cv2.filter2D(self.image, cv2.CV_8UC1, gauss_kernel)

        kernel_l = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], np.float32)
        self.image = cv2.filter2D(self.image, cv2.CV_8UC1, kernel_l)

        ret, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.image_states.append(self.image)
        return self.image
