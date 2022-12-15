import easyocr
from operator import itemgetter
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import image_process

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DetectorRes:
    img: np.ndarray
    bbox: BBox


def detect(img: np.ndarray,
           kernel_size: int,
           sigma: float,
           theta: float,
           min_area: int) -> List[DetectorRes]:
    """
    Technique de segmentation de mots propose par R. Manmatha
    http://ciir.cs.umass.edu/pubfiles/mm-27.pdf.

    Args:
        img: A grayscale uint8 image.
        kernel_size: The size of the filter kernel, must be an odd integer.
        sigma: Standard deviation of Gaussian function used for filter kernel.
        theta: Approximated width/height ratio of words, filter function is distorted by this factor.
        min_area: Ignore word candidates smaller than specified area.

    Returns:
        List of DetectorRes instances, each containing the bounding box and the word image.
    """
    assert img.ndim == 2
    assert img.dtype == np.uint8

    # apply filter kernel
    kernel = _compute_kernel(kernel_size, sigma, theta) # creation du kernel qu'on va appliquer
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    img_thres = 255 - cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # append components to result
    res = []
    components = cv2.findContours(img_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < min_area:
            continue
        # append bounding box and image of word to result list
        x, y, w, h = cv2.boundingRect(c)  # bounding box as tuple (x, y, w, h)
        crop = img[y:y + h, x:x + w]
        res.append(DetectorRes(crop, BBox(x, y, w, h)))

    return res


def _compute_kernel(kernel_size: int,
                    sigma: float,
                    theta: float) -> np.ndarray:

    """ On va creer un filtre gaussien anisotropic qui va etre notre kernel pour appliquer le filtrage sur l'image"""

    assert kernel_size % 2  # doit etre un nombre impair

    # create coordinate grid
    half_size = kernel_size // 2
    xs = ys = np.linspace(-half_size, half_size, kernel_size)
    x, y = np.meshgrid(xs, ys)

    # compute sigma values in x and y direction, where theta is roughly the average x/y ratio of words
    sigma_y = sigma
    sigma_x = sigma_y * theta

    # compute terms and combine them
    exp_term = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
    x_term = (x ** 2 - sigma_x ** 2) / (2 * np.math.pi * sigma_x ** 5 * sigma_y)
    y_term = (y ** 2 - sigma_y ** 2) / (2 * np.math.pi * sigma_y ** 5 * sigma_x)
    kernel = (x_term + y_term) * exp_term

    # normalize and return kernel
    kernel = kernel / np.sum(kernel)
    return kernel


def prepare_img(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale image (if needed) and resize to given height."""
    assert img.ndim in (2, 3)
    assert img.dtype == np.uint8
    if img.ndim == 3:
        img = image_process.renforce_contours(img, 170)

    return img


def _cluster_lines(detections: List[DetectorRes],
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    # compute matrix containing Jaccard distances (which is a proper metric)
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(i, num_bboxes):
            a = detections[i].bbox
            b = detections[j].bbox
            if a.y > b.y + b.h or b.y > a.y + a.h:
                continue
            intersection = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            union = a.h + b.h - intersection
            iu = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = dist_mat[j, i] = 1 - iu  # Jaccard distance is defined as 1-iu

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])

    res = sorted(clustered.values(), key=lambda line: [det.bbox.y + det.bbox.h / 2 for det in line])
    return res


def sort_multiline(detections: List[DetectorRes],
                   max_dist: float = 10,
                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    """Cluster detections into lines, then sort the lines according to x-coordinates of word centers.

    Args:
        detections: List of detections.
        max_dist: Maximum Jaccard distance (0..1) between two y-projected words to be considered as neighbors.
        min_words_per_line: If a line contains less words than specified, it is ignored.

    Returns:
        List of lines, each line itself a list of detections.
    """
    lines = _cluster_lines(detections, max_dist, min_words_per_line)
    res = []
    for line in lines:
        res += sort_line(line)
    return res


def sort_line(detections: List[DetectorRes]) -> List[List[DetectorRes]]:
    """
    On calcule le centre de chaque box sur le sens de la largeur et on trie par rapport
    à ca
    """
    return [sorted(detections, key=lambda det: det.bbox.x + det.bbox.w / 2)]


'''
image_reader = easyocr.Reader(['en'])


def detect_text_regions(img):
    result = []
    ligne = []
    final_result = []

    img = image_process.renforce_contours(img, 170)
    output = image_reader.readtext(img)

    for zone in output:
        cord = zone[0]
        xmin, ymin = [min(idx) for idx in zip(*cord)]
        xmax, ymax = [max(idx) for idx in zip(*cord)]
        text_zone_cord = [xmin, ymin, xmax, ymax]
        inserted = False
        
        Les textes detectes ne sont pas ordonnees, donc le but une fois les zones de texte detectes
        est de les agencer de sorte a avoir en premier les lignes puis dans chaque ligne trier par
        rapport au texte qui vient en premier 
        

        # apres ce traitement les cooordonnes sont ajoutes au resultat final tries par ligne
        if len(result) == 0:
            result.insert(0, text_zone_cord)
        else:
            for i in range(len(result)):
                if ymin <= result[i][1]:
                    result.insert(i, text_zone_cord)
                    inserted = True
                    break
            if not inserted:
                result.append(text_zone_cord)
        
        on va maintenant essayer de constituer chaque ligne et faire un tri dans chaque ligne
        pour constituer le resultat final
        
    ligne.append(result[0])
    for i in range(1, len(result)):
        if result[i][1] - result[i - 1][1] < 20:
            ligne.append(result[i])
        else:
            final_result.append(sorted(ligne, key=itemgetter(0)))
            ligne = [result[i]]

    final_result.append(ligne)

    return final_result

'''
