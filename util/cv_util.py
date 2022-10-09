import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas as mt
from scipy.signal import find_peaks


def cv_imread(file_path,color_patten=cv2.IMREAD_COLOR):
    """
    读取中文路径图片
    :ctime: 2022.06.20
    :param file_path:图片目录
    :param color_patten: 图片模式
    :return: 对应模式下的图片矩阵
    """
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),color_patten)
    return cv_img

def cv_imwrite(file_path,image, color_patten=cv2.COLOR_BGR2RGB, prefix='.bmp'):
    """
    写入中文路径图片
    :ctime: 2022.07.12
    :param file_path:图片目录
    :param color_patten: 图片模式
    :param prefix: 图片格式
    :return: 对应模式下的图片矩阵
    """
    if len(image.shape) == 3: # 彩色图，三维数据
        cv_img = cv2.imencode(prefix, cv2.cvtColor(image, color_patten))[1].tofile(file_path) # 保存带有中文路径的图片

    elif len(image.shape) == 2: # 灰度图，一维数据
        #cv_img_tmp = cv2.imencode(prefix, image)[1].tofile(file_path)
        cv_img = cv2.imencode(prefix, image)[1].tofile(file_path) # 保存带有中文路径的图片

    else:
        assert('wrong image dims!')
        return

    return cv_img

def cv_resize(img, scale_ratio):
    """
    修改图片尺寸
    :ctime: 2022.06.21
    :param img: 原始图片
    :param scale_ratio: 缩放比例
    :return: 缩放后的图片
    """
    width = int(img.shape[1] * scale_ratio)
    height = int(img.shape[0] * scale_ratio)

    img_dsize = cv2.resize(img, (width, height))

    return img_dsize

def computeMean(image, ignore_zeros=False):
    """
    计算灰度化图片像素平均值
    :param image: 灰度化图像
    :param ignore_zeros: 是否剔除值为0的像素点
    :return: 像素平均灰度值
    """
    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
    else:
        select = np.ravel

    pixels = select(image)

    return np.mean(pixels)

def get_min(image, ignore_zeros=False):
    """
    计算灰度化图片像素平均值
    :param image: 灰度化图像
    :param ignore_zeros: 是否剔除值为0的像素点
    :return: 像素平均灰度值
    """
    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
    else:
        select = np.ravel

    pixels = select(image)

    return np.min(pixels)

def cv_obtain_des(img):
    """
    剔除背景，提取目标本身
    :param img: 原始图像
    :return img_des: 剔除背景后的图像
    """
    # 高斯降噪
    img_blured = cv2.GaussianBlur(img, (7, 7), 0)
    #
    # 灰度图
    img_grey = cv2.cvtColor(img_blured, cv2.COLOR_RGB2GRAY)

    # 二值化图像
    dst, img_bin = cv2.threshold(img_grey, 29, 255, cv2.THRESH_BINARY)  # 自动计算分割阈值
    img_bin_3 = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    # 提取目标本身
    img_fin = cv2.bitwise_and(img, img_bin_3) # 原始图像与目标掩码取交
    # img_des = cv2.cvtColor(img_fin, cv2.COLOR_BGR2RGB)

    return img_fin

def cv_kmeans(image_src, n_cluster=3):
    """
    对图像进行聚类
    :param image_src: 原始图像：彩色图或灰度图
    :param n_cluster: 类别数量
    :return: segmented_image - 聚类后的图像
             segmented_areas - 各个类别的面积
    """
    image = image_src.copy()

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    if len(image.shape) == 3: # 彩色图，三维数据
        pixel_values = image.reshape((-1, 3))
    elif len(image.shape) == 2: # 灰度图，一维数据
        pixel_values = image.reshape((-1, 1))
    else:
        assert('wrong image dims!')
        return

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    # 聚成三类：背景、烟叶主色、杂色
    _, labels, (centers) = cv2.kmeans(pixel_values, n_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # 各类像素点个数（面积）
    segmented_list = np.unique(segmented_image).tolist()
    segmented_areas = [len((np.where(segmented_image == segmented_list[v]))[0]) for v in range(0, len(segmented_list))]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # show the image
    #plt_imshow(segmented_image)

    return segmented_image, segmented_areas, centers.tolist()


def plt_imshow(image, figszie=(15,7)):
    plt.figure(figsize=figszie)
    if len(image.shape) == 3: # 彩色图，三维数据
        plt.imshow(image)
    elif len(image.shape) == 2: # 灰度图，一维数据
        plt.imshow(image,cmap='gray')
    else:
        assert('wrong image dims!')
        return

def cv_drawContours(img_src, img_bin, area_thresh):
    """
    寻找图像轮廓，并设置面积阈值标记轮廓
    :param img_src:
    :param img_bin:
    :param area_thresh:
    :return:
    """
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_thresh:  # 大轮廓，根据此阈值可以找到烟叶内部的孔洞
            cv_contours.append(contour)

    # 4、标出烟叶内部孔洞区域
    cv_contours.sort(key=lambda i: len(i), reverse=True)
    inner_contour = cv_contours#[1:]  # 剔除最大轮廓（仅标记内部轮廓）
    for contour in inner_contour:
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(img_src, [contour], -1, (255, 25, 255), 4)

    plt.figure()
    plt.imshow(img_src)

    return inner_contour


# Gabor filter kernel initialization
def init_filters():
    DIVANGLES = 6
    filters = []

    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / DIVANGLES):
        kern = cv2 .getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype = cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters #Note

# string drawing
def drawNote(tileimg, x, y, width, theta):
    fontsize = 0.4
    ds ='theta =% d deg'% (theta)
    ssize, basel = cv2.getTextSize(ds, cv2.FONT_HERSHEY_SIMPLEX, fontsize, 1)
    print(ssize, basel)

    dx = x + width-ssize[0]
    dy = y + ssize[1] +2
    cv2.rectangle(tileimg, (dx, dy-ssize [1]), (dx + ssize [0], dy + basel), (255, 255, 255), -1)
    cv2.putText(tileimg, ds, (dx, dy),
            cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,0,0), 1, cv2.LINE_AA)

# Filtering
def filterProcess(img, filters, SCALE = 0.75):
    DIVANGLES = 6
    height, width = img.shape[:2]
    height = int(height * SCALE)
    width = int(width * SCALE)

    if len(img.shape) == 2:
        plane = 1 # number of image channels = 1
        charcolor = (0,0,0)
        filterType = cv2.CV_8UC1
    else:
        plane = img.shape[2] # Number of image channels = 3
        charcolor = (255)
        filterType = cv2.CV_8UC3
    tileimg = np.zeros((height * 3, width * 2, 3), np.uint8) #Tiling image

    accum = np.zeros_like(img)
    for kern, theta, i in zip(filters, np.arange(0, 180, 180 / DIVANGLES), range(0, DIVANGLES)):
        fimg = cv2.filter2D(img, filterType, kern)

        # reduced, tile image
        x = (i % 2) * width
        y = (i // 2) * height
        rimg = cv2.resize(fimg, (width, height), interpolation = cv2.INTER_CUBIC)
        if plane == 1:
            tileimg[y: y + height, x: x + width] = cv2.merge((rimg, rimg, rimg))
        else:
            tileimg[y: y + height, x: x + width] = rimg

        # Note string drawing
        drawNote(tileimg, x, y, width, theta) #maximum

        np.maximum(accum, fimg, accum)
    return accum, tileimg

# Gabor filter processing
def gaborFilter(img):
    """

    :param img:
    :return:
    """
    filters = init_filters()
    resimg, tileimg = filterProcess(img, filters)

    # # Display
    # cv2.imshow('result', resimg)
    # cv2.imshow('tileimg', tileimg)
    # cv2.waitKey()

    return resimg, tileimg

def extract_texture_hara(img):
    textures = mt.features.haralick(img, ignore_zeros=True, compute_14th_feature=False)
    ht_mean = textures.mean(axis=0)

    return ht_mean

def extract_texture_lbp(img):
    textures, img_lbp, trans_time = mt.features.lbp(img, radius=3, points=16, ignore_zeros=False)

    return textures, img_lbp, trans_time

def obtain_textures(image_src):
    # 0、cv2读取图像采用BGR模式，matplotlib读取图像采用RGB模式，
    image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    img = image.copy()

    # 1、原始图片高斯降噪，R、G、B三通道加权计算为单通道的灰度图
    img_blured = cv2.GaussianBlur(image, (7, 7), 0)
    img_gray = cv2.cvtColor(img_blured, cv2.COLOR_RGB2GRAY)
    # plt.figure(3)
    # cv_hist(img_gray)
    #
    # plt.figure(4)
    # plt.imshow(img_gray,cmap='gray')
    # plt.show()

    # 2、根据剔除阈值，初步提取二值化图像
    dst, img_bin = cv2.threshold(img_gray, 29, 255, cv2.THRESH_BINARY)  # 设置分割阈值，根据灰度图像素值分布图设置
    # plt.figure(5)
    # plt.imshow(img_bin)
    # plt.show()

    # 3、连通区域，移除小面积区域
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(img_bin)
    statsList = stats.tolist()
    statsList.sort(key=lambda i: i[4], reverse=True)
    for stat in statsList[2:]:  # 保留最大连通区域，其他连通区域填充为0
        cv2.rectangle(img_bin, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (0, 0, 0),
                      -1)

    # 4、提取轮廓
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda i: len(i), reverse=True)

    inner_contour = contours[1:]
    for contour in inner_contour:
        cv2.drawContours(img_bin, [contour], -1, (255, 255, 255), -1)

    # 5、提取目标本身
    img_bin_3 = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    img_des = cv2.bitwise_and(img, img_bin_3)  # 原始图像与目标掩码取交

    # plt.figure(10)
    # cv_hist(img_des)
    #
    # plt.figure(11)
    # plt.imshow(img_des)
    # plt.show()

    # 6、提取烟叶纹理
    img_des_gray = cv2.cvtColor(img_des, cv2.COLOR_RGB2GRAY)
    img_garbor, _ = gaborFilter(img_des_gray)

    #img_texture = extract_texture_hara(img_garbor)
    img_texture = extract_texture_lbp(img_garbor)

    return img_texture, img_des

def zh_ch(string):
    return string.encode('gbk').decode(errors='ignore')

def cv_hist(img, figszie=(15,7)):
    plt.figure(figsize=figszie)
    dims = img.ndim
    if dims == 3:
        ## 彩色图
        # calcHist
        # 参数1：要计算的原图，以方括号的传入，如：[img]。
        # 参数2：类似前面提到的dims，灰度图写[0]
        # 就行，彩色图R / G / B分别传入[0] / [1] / [2]。
        # 参数3：要计算的区域ROI，计算整幅图的话，写None。
        # 参数4：也叫bins, 子区段数目，如果我们统计0 - 255
        # 每个像素值，bins = 256；如果划分区间，比如0 - 15, 16 - 31…240 - 255
        # 这样16个区间，bins = 16。
        # 参数5：range, 要计算的像素值范围，一般为[0, 256)。
        color = ('r', 'g', 'b')
        hists = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=col)
            plt.xlim([0, 256])
            hists.append(hist)
        plt.legend(loc='best')
        #plt.show()

        return hists
    elif dims == 2:
        ## 灰度图
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        #plt.show()

        return hist
    else:
        return

def get_peaks(srcArray,obtainNum=3):
    """
    画出数组直方图，寻找极值点
    :param srcArray: 原始一维数组
    :param obtainNum: 极值点数量
    :return: 找到极值点，并在直方图中标出
    """
    peakIdxs, _ = find_peaks(srcArray, threshold=0)
    peakVals = srcArray[peakIdxs]

    # 在极值点列表中的排序
    idxinpeakIdxs = peakVals.argsort()[::-1][0:obtainNum]

    # 在原始数组中的排序
    obtainIdxs = peakIdxs[idxinpeakIdxs]

    plt.plot(srcArray)
    plt.plot(obtainIdxs, srcArray[obtainIdxs], "x")
    # for i, j in zip(obtainIdxs, srcArray[obtainIdxs]):
    #     plt.text(i, j, '(%s,%s)' % (i, j), family='monospace', fontsize=12, color='r')
    plt.stem(obtainIdxs, srcArray[obtainIdxs])
    plt.xlim([0, 256])
    plt.show()