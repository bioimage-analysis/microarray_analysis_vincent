import numpy as np
import matplotlib.pyplot as plt
from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage import util, filters, morphology, exposure, measure
from skimage.draw import circle
import pandas as pd
from ipywidgets import interact, fixed, interactive
from IPython.display import display
import holoviews as hv
import holoviews.util
from holoviews import streams
from bokeh.models import HoverTool
from skimage.color import label2rgb


def crop_show(image, size = 7000, save = False):
    x, y = image.shape
    x_crop = np.int((x-size)/2)
    y_crop = np.int((y-size)/2)


    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(image)
    ax.plot([y_crop,y_crop+size],[x_crop,x_crop], [y_crop,y_crop],[x_crop,x_crop+size],
              [y_crop,y_crop+size],[x_crop+size,x_crop+size], [y_crop+size,y_crop+size],[x_crop,x_crop+size],
              'k-', color='r')

    if save == True:
        fig.savefig('cropped_region.png', dpi =300)

    return image[x_crop:x_crop+size, y_crop:y_crop+size]

def crop(image, size = 7000):
    x, y = image.shape
    x_crop = np.int((x-size)/2)
    y_crop = np.int((y-size)/2)

    return image[x_crop:x_crop+size, y_crop:y_crop+size]


def chunks(img, overlap = 100, chunks = 8):

    x0, y0 = img.shape

    x_block, y_block = np.int((x0 + overlap)/chunks), np.int((y0 + overlap)/chunks)
    step = x_block - overlap
    chunks_img = util.view_as_windows(img, (x_block,y_block), step = step)
    chunks_img = np.squeeze(chunks_img)

    return chunks_img, x_block, y_block, overlap


def canny_chunks(chunks):
    lst = []
    for i in range(chunks.shape[0]):
        for j in range(chunks.shape[1]):
            lst.append(canny(chunks[i,j], sigma=1))
    return(lst)

def hough_circle_chunk(lst_canny, low = 17, high = 25):
    lst = []
    for k in range(len(lst_canny)):
        # Shoould be able to change the range:
        hough_radii = np.arange(low, high, 1)
        #hough_radii = np.asarray([20])
        hough_res = hough_circle(lst_canny[k], hough_radii)

        # Select the most prominent 5 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                min_xdistance=50, min_ydistance=50)
        lst.append((accums, cx, cy, radii))
    return(lst)

def draw_circle(lst_canny, lst_hough):
    x,y = lst_canny[0].shape
    lst = []
    for l in range(len(lst_canny)):
        img_mask = np.zeros((x+30,y + 30), dtype="bool")
        _, cx, cy, _ = lst_hough[l]
        for center_y, center_x in zip(cy, cx):
            circy, circx = circle(center_y, center_x, 20)
            img_mask[circy, circx] = True
        lst.append(img_mask)
    return(lst)


def stitched_x(lst_canny, list_mask, over = 200):
    #over should be equla 1/2 overlay
    x,y = lst_canny[0].shape
    stitched_x = np.hstack((list_mask[0][:, 0:y-over], list_mask[1][:, over:y-over], list_mask[2][:, over:y-over],
                            list_mask[3][:, over:y-over], list_mask[4][:, over:y-over], list_mask[5][:, over:y-over],
                            list_mask[6][:, over:y-over], list_mask[7][:, over:y-over], list_mask[8][:, over:y]))
    return(stitched_x)

def stitched_y(list_mask, over = 200):
    x, y = list_mask[0].shape
    stitched_y = np.vstack((list_mask[0][0:x-over, :], list_mask[1][over:x-over, :], list_mask[2][over:x-over, :],
                            list_mask[3][over:x-over, :], list_mask[4][over:x-over, :], list_mask[5][over:x-over, :],
                            list_mask[6][over:x-over, :], list_mask[7][over:x-over, :], list_mask[8][over:x, :]))
    return(stitched_y)

def stitched(lst_canny, list_mask):

    lst_stitched_x = []
    for x in range(0, len(list_mask), 9):
        lst_stitched_x.append(stitched_x(lst_canny,list_mask[x:len(list_mask)], over = 100))

    stitched_img = stitched_y(lst_stitched_x, over = 115)

    return(measure.label(stitched_img, background = 0))

def bkg_correct(img):
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(img, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    correct = img - bkg.background
    correct = correct+np.abs(correct.min())
    return(correct)

def prop_measure(labeled_img, intensity_image, back_sub = False):

    if back_sub:
        intensity_image = bkg_correct(intensity_image)
    #only use the labeled region
    props = measure.regionprops(labeled_img, intensity_image=intensity_image)

    mean_lst = [np.mean(prop.intensity_image[np.nonzero(prop.intensity_image)]) for prop in props]
    median_lst = [np.median(prop.intensity_image[np.nonzero(prop.intensity_image)]) for prop in props]
    std_lst = [np.std(prop.intensity_image[np.nonzero(prop.intensity_image)]) for prop in props]
    percentile_75_lst = [np.percentile(prop.intensity_image[np.nonzero(prop.intensity_image)], 75) for prop in props]
    percentile_50_lst = [np.percentile(prop.intensity_image[np.nonzero(prop.intensity_image)], 50) for prop in props]
    percentile_25_lst = [np.percentile(prop.intensity_image[np.nonzero(prop.intensity_image)], 25) for prop in props]

    property_lst = [mean_lst, median_lst, std_lst, percentile_75_lst,
                    percentile_50_lst, percentile_25_lst, props]
    return property_lst, props

def data_panda(property_lst):


    data = {"mean intensity": property_lst[0],
               "median intensity": property_lst[1],
               "standard deviation": property_lst[2],
               "25th percentile ": property_lst[3],
               "50th percentile ": property_lst[4],
               "75th percentile ": property_lst[5]}

    return(pd.DataFrame(data))


def new_label_holo(liste, labeled_img, int_img):

    new_labeled_img = np.copy(labeled_img)
    for roi in liste:
        xx, yy = np.meshgrid(np.arange(int(roi[3]),int(roi[1])), np.arange(int(roi[0]),int(roi[2])))
        new_labeled_img[xx, yy] = 0

    props = measure.regionprops(new_labeled_img,
        intensity_image=int_img)

    circularity = [prop.eccentricity for prop in props]

    new_labels = new_labeled_img.copy()
    for circ, prop in zip(circularity, props):
        if circ >=0.7:
            new_labels[tuple(prop.coords.T)] = 0
        else:
            new_labels[tuple(prop.coords.T)] = prop.label
    return(new_labels)


def prop_lab(labeled_img):
    props = measure.regionprops(labeled_img)
    lst_coord = [prop.centroid[::-1] for prop in props]
    label = [prop.label for prop in props]
    data = np.concatenate((np.asarray(lst_coord),
                           np.zeros(np.asarray(label)[:,np.newaxis].shape)),
                           axis = 1)
    data = {"x": data[:,0],
            "y": data[:,1],
            "empty": data[:,2]}
    return data


def equaliz(FAM_cropped):
    # Need to crop a little more after stitching
	FAM_cropped_n = FAM_cropped[0:6530, 0:6500]
	to_show_FAM = FAM_cropped_n[::-1]
	p2, p98 = np.percentile(to_show_FAM, (2, 98))
	equal = exposure.rescale_intensity(to_show_FAM, in_range=(p2, p98))
	return equal, FAM_cropped_n

def to_ROI(to_show_FAM, data):
    hv.output(size=200)
    dict_spec = {'Points':{'style':dict(cmap='gray', size=0.1, alpha=0.1),
                           'plot':dict(color_index=2, colorbar=True ,invert_yaxis=True, toolbar='above')},
                 'Image':{'style':dict(cmap= 'gray'),
                          'plot':dict(invert_yaxis=True)}}


    image = hv.Image(to_show_FAM[::8,::8], bounds=(0,0,to_show_FAM.shape[1],to_show_FAM.shape[0]), label= "FAM")
    label = hv.Points(data, vdims=['empty'])

    box = streams.Bounds(source=label, bounds=(0,0,0,0))
    bounds = hv.DynamicMap(lambda bounds: hv.Bounds(bounds), streams=[box])

    return image, label, box, bounds, dict_spec

def show_RGB(labeled_img, s = 4):
    hv.output(size=200)
    dict_spec = {'RGB':{'plot':dict(xaxis=None, yaxis=None)}}

    rescaled = labeled_img[::s,::s]
    to_show = hv.RGB(label2rgb(rescaled, bg_label=0))
    return to_show.opts(dict_spec)

def show_img(labeled_img, s = 4):
    # Need to flip image
    to_show = labeled_img[::-1]

    hv.output(size=200)
    dict_spec = {'Image':{'style':dict(cmap= 'nipy_spectral'),
                          'plot':dict(invert_yaxis=True)}}
    display = hv.Image(to_show[::s,::s], bounds=(0,0,to_show.shape[1],to_show.shape[0]))
    return display.opts(dict_spec)

def dict_for_plot(property_lst, props):

    lst_coord = [prop.centroid[::-1] for prop in props]

    data = np.concatenate((np.asarray(lst_coord),
                           np.asarray(property_lst[2])[:,np.newaxis],
                           np.asarray(property_lst[0])[:,np.newaxis],
                           np.asarray(property_lst[1])[:,np.newaxis],
                           np.asarray(property_lst[3])[:,np.newaxis],
                           np.asarray(property_lst[4])[:,np.newaxis],
                           np.asarray(property_lst[5])[:,np.newaxis]),
                           axis = 1)
    data = {"x": data[:,0],
            "y": data[:,1],
            "std": data[:,2],
            "mean": data[:,3],
            "median": data[:,4],
            "Q1": data[:,5],
            "Q2": data[:,6],
            "Q3": data[:,7]}
    return data


def create_hover():
    hover1 = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x{0000}, $y{0000})"),
            ("std", "@std"),
                 ]
                    )

    hover2 = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x{0000}, $y{0000})"),
            ("mean", "@mean"),
                ]
                    )

    hover3 = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x{0000}, $y{0000})"),
            ("median", "@median"),
                ]
                    )
    hover4 = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x{0000}, $y{0000})"),
            ("Q1", "@Q1"),
                ]
                    )
    hover5 = HoverTool(
        tooltips=[
            ("index", "$index"),
           ("(x,y)", "($x{0000}, $y{0000})"),
            ("Q2", "@Q2"),
                ]
                    )
    hover6 = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x{0000}, $y{0000})"),
            ("Q3", "@Q3"),
                ]
                    )

    return [hover1, hover2, hover3, hover4, hover5, hover6]


def plot_result(property_lst, props, to_show_FAM, dye = "FAM"):
    data = dict_for_plot(property_lst, props)
    hover_lst = create_hover()

    hv.output(size=200)
    dict_spec = {'Points':{'style':dict(cmap='viridis', size=8),
                           'plot':dict(color_index=2, colorbar=True ,invert_yaxis=True, toolbar='above')},
                 'Image':{'style':dict(cmap= 'gray'),
                          'plot':dict(invert_yaxis=True)},
                 'Overlay':{'plot':dict(tabs=True)}}

    image = hv.Image(to_show_FAM[::8,::8], bounds=(0,0,to_show_FAM.shape[1],to_show_FAM.shape[0]), label= dye)

    std = hv.Points(data, vdims=['std'], label = 'std').opts(plot=dict(tools=[hover_lst[0]]))
    mean = hv.Points(data, vdims=['mean'], label = 'mean').opts(plot=dict(tools=[hover_lst[1]]))
    median = hv.Points(data, vdims=['median'], label = 'median').opts(plot=dict(tools=[hover_lst[2]]))
    Q1 = hv.Points(data, vdims=['Q1'], label = 'Q1').opts(plot=dict(tools=[hover_lst[3]]))
    Q2 = hv.Points(data, vdims=['Q2'], label = 'Q2').opts(plot=dict(tools=[hover_lst[4]]))
    Q3 = hv.Points(data, vdims=['Q3'], label = 'Q3').opts(plot=dict(tools=[hover_lst[5]]))

    dlayout = image * std * mean * median * Q1 * Q2 * Q3

    return dlayout.opts(dict_spec)





'''
def coord_roi_exclude(w):
    start, end = w.result
    xx, yy = np.meshgrid(np.arange(start[0],end[0]), np.arange(start[1],end[1]))
    return np.transpose(np.vstack([xx.ravel(), yy.ravel()]))

def lst_label_in_roitoexclude(propregion_all, positions_toexclude):
    coords = [(prop.coords, prop.label) for prop in propregion_all]

    lst_label = []
    for coord in coords:
        A = np.in1d(coord[0][:,0], positions_toexclude[:,0])
        B = np.in1d(coord[0][:,1], positions_toexclude[:,1])
        C = np.stack((A,B))
        if np.any(np.all(C == True,axis=0)):
            lst_label.append(coord[1])
    return(np.asarray(mean_lst_label, dtype = int))


            mean_labels_bis[tuple(region.coords.T)] = mean[0]
'''
