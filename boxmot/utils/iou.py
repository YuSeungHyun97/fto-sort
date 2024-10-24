# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np

def iou_batch(bboxes1, bboxes2) -> np.ndarray:
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
        (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
        wh
    )
    return o

def expand_bboxes(bboxes, scale=0.0):
    """
    Expands the bounding boxes by a given scale.
    """
    # Convert to numpy array if input is a list
    bboxes = np.array(bboxes)

    # Calculate width and height
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    
    # Calculate the center points
    centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
    centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
    
    # Expand width and height by scale
    new_widths = widths * (1 - scale)
    new_heights = heights * (1 - scale)
    
    # Calculate new bounding boxes
    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0] = centers_x - new_widths / 2  # new x1
    new_bboxes[:, 1] = centers_y - new_heights / 2  # new y1
    new_bboxes[:, 2] = centers_x + new_widths / 2  # new x2
    new_bboxes[:, 3] = centers_y + new_heights / 2  # new y2
    
    return new_bboxes

def biou_batch(bboxes1, bboxes2, buffer_scale=0.03):
    """
    Computes IoU between two sets of bboxes after expanding them by a given scale.
    """
    # Expand bounding boxes
    bboxes1_expanded = expand_bboxes(bboxes1, buffer_scale)
    bboxes2_expanded = expand_bboxes(bboxes2, buffer_scale)
    
    # IoU computation
    bboxes2_expanded = np.expand_dims(bboxes2_expanded, 0)
    bboxes1_expanded = np.expand_dims(bboxes1_expanded, 1)

    xx1 = np.maximum(bboxes1_expanded[..., 0], bboxes2_expanded[..., 0])
    yy1 = np.maximum(bboxes1_expanded[..., 1], bboxes2_expanded[..., 1])
    xx2 = np.minimum(bboxes1_expanded[..., 2], bboxes2_expanded[..., 2])
    yy2 = np.minimum(bboxes1_expanded[..., 3], bboxes2_expanded[..., 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1_expanded[..., 2] - bboxes1_expanded[..., 0]) * 
        (bboxes1_expanded[..., 3] - bboxes1_expanded[..., 1]) +
        (bboxes2_expanded[..., 2] - bboxes2_expanded[..., 0]) * 
        (bboxes2_expanded[..., 3] - bboxes2_expanded[..., 1]) -
        wh
    )
    return o

def giou_batch(bboxes1, bboxes2) -> np.ndarray:
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
        (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
        wh
    )

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.0) / 2.0  # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2) -> np.ndarray:
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
        (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
        wh
    )

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ciou_batch(bboxes1, bboxes2) -> np.ndarray:
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
        (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
        wh
    )

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.0
    h1 = h1 + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi**2)) * (arctan**2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v

    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)


def centroid_batch(bboxes1, bboxes2, w, h) -> np.ndarray:
    """
    Computes the normalized centroid distance between two sets of bounding boxes.
    Bounding boxes are in the format [x1, y1, x2, y2].
    `normalize_scale` is a tuple (width, height) to normalize the distance.
    """

    # Calculate centroids
    centroids1 = np.stack(((bboxes1[..., 0] + bboxes1[..., 2]) / 2,
                           (bboxes1[..., 1] + bboxes1[..., 3]) / 2), axis=-1)
    centroids2 = np.stack(((bboxes2[..., 0] + bboxes2[..., 2]) / 2,
                           (bboxes2[..., 1] + bboxes2[..., 3]) / 2), axis=-1)

    # Expand dimensions for broadcasting
    centroids1 = np.expand_dims(centroids1, 1)
    centroids2 = np.expand_dims(centroids2, 0)

    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))

    # Normalize distances
    norm_factor = np.sqrt(w**2 + h**2)
    normalized_distances = distances / norm_factor

    return 1 - normalized_distances


def run_asso_func(func, *args):
    """
    Wrapper function that checks the inputs to the association functions
    and then call either one of the iou association functions or centroid.

    Parameters:
    func: The batch function to call (either *iou*_batch or centroid_batch).
    *args: Variable length argument list, containing either bounding boxes and optionally size parameters.
    """
    if func not in [iou_batch, biou_batch, giou_batch, diou_batch, ciou_batch, centroid_batch]:
        raise ValueError("Invalid function specified. Must be either '(g,d,c, )iou_batch' or 'centroid_batch'.")

    if func in (iou_batch, biou_batch, giou_batch, diou_batch, ciou_batch):
        if len(args) != 4 or not all(isinstance(arg, (list, np.ndarray)) for arg in args[0:2]):
            raise ValueError("Invalid arguments for iou_batch. Expected two bounding boxes.")
        return func(*args[0:2])
    elif func is centroid_batch:
        if len(args) != 4 or not all(isinstance(arg, (list, np.ndarray)) for arg in args[:2]) or not all(isinstance(arg, (int)) for arg in args[2:]):
            raise ValueError("Invalid arguments for centroid_batch. Expected two bounding boxes and two size parameters.")
        return func(*args)
    else:
        raise ValueError("No such association method")


def get_asso_func(asso_mode):
    ASSO_FUNCS = {
        "iou": iou_batch,
        "biou": biou_batch,
        "giou": giou_batch,
        "ciou": ciou_batch,
        "diou": diou_batch,
        "centroid": centroid_batch
    }

    return ASSO_FUNCS[asso_mode]
