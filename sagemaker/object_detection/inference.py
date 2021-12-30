import cv2
import numpy as np
import time
from openvino.inference_engine import IECore


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(boxes, scores, iou_thresh):
    """
    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_thresh (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组, ::-1表示逆序
    order = scores.argsort()[::-1]

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= iou_thresh)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]

    return np.array(temp)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf = x[:, 5:].max(axis=1, keepdims=True)
        j = x[:, 5:].argmax(axis=1)
        j = np.expand_dims(j, axis=-1)

        x = np.concatenate((box, conf, j), axis=-1)
        x = x[np.where(conf[:, 0] > conf_thres)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


if __name__ == "__main__":
    ie = IECore()

    net = ie.read_network(
        model="./best_openvino_model/best.xml",
        weights="./best_openvino_model/best.bin",
    )
    exec_net = ie.load_network(net, "CPU")

    output_layer_ir = next(iter(exec_net.outputs))
    input_layer_ir = next(iter(exec_net.input_info))

    # load an image
    # Text detection models expects image in BGR format
    image = cv2.imread("./IMG_00000.jpg")

    # N,C,H,W = batch size, number of channels, height, width
    N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims

    # Resize image to meet network expected input sizes
    # resized_image = cv2.resize(image, (W, H))
    im, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640), stride=32, auto=False)

    # Reshape to network input shape
    input_image = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    input_image = np.expand_dims(input_image, 0)
    input_image = input_image / 255.0

    # do inference
    iter_times = 50
    t1 = time.time()
    for _ in range(iter_times):
        result = exec_net.infer(inputs={input_layer_ir: input_image})
    t2 = time.time()
    time_cost = (t2 - t1) / iter_times
    print("Time cost each frame (640 x 640) = {} ms".format(1000 * time_cost))

    pred = result["output"]
    conf_thres = 0.40
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    class_names = [
        "2D_CODE", "Caution", "3C", "EAC", "UL2", "WEEE", "KC",
        "ATEX", "FM2", "Failsafe", "RCM", "FM", "CE", "UL1"]

    colors = {
        "2D_CODE": (255, 51, 51),
        "Caution": (255, 0, 255),
        "3C": (255, 128, 0),
        "EAC": (0, 153, 0),
        "UL2": (200, 153, 89),
        "WEEE": (10, 29, 199),
        "KC": (49, 48, 153),
        "ATEX": (40, 210, 144),
        "FM2": (182, 255, 40),
        "Failsafe": (100, 92, 49),
        "RCM": (255, 80, 153),
        "FM": (255, 20, 20),
        "CE": (190, 153, 153),
        "UL1": (100, 153, 153),
    }

    full_path = "./IMG_00000.jpg"
    vis_image = cv2.imread(full_path, cv2.IMREAD_COLOR)
    height, width, channels = vis_image.shape

    scale_w, scale_h = ratio

    detections = pred[0]

    for det in detections:
        x_min = (det[0] - dw) / scale_w
        y_min = (det[1] - dh) / scale_h
        x_max = (det[2] - dw) / scale_w
        y_max = (det[3] - dh) / scale_h

        color = colors[class_names[int(det[-1])]]
        color = (color[2], color[1], color[0])
        label_info = '{} {:.3f}'.format(class_names[int(det[5])], det[4])
        vis_image = cv2.rectangle(vis_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 5)
        vis_image = cv2.putText(vis_image, label_info, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color,
                                2, cv2.LINE_AA)

    cv2.imwrite("./result.png", vis_image)
