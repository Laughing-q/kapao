import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import argparse
import numpy as np
import os.path as osp
import os
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size
from yolov5.utils.plots import plot_one_box
from utils.datasets import LoadImages
from models.experimental import attempt_load
import torch
import cv2
import yaml
from tqdm import tqdm
from val import run_nms, post_process_batch
from trackers.tracker import ObjectTracker
from trackers.counter import TrailParser


TAG_RES = {135: "480p", 136: "720p", 137: "1080p"}


def point_in_areas(pt, areas, frame=None):
    stats = []
    for area in areas:
        x1, y1, x2, y2 = area
        lt = (pt[0] > x1) and (pt[1] > y1)
        rb = (pt[0] < x2) and (pt[1] < y2)
        stats.append(lt and rb)
        if frame is not None:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)

    return any(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # video options
    parser.add_argument("--source", default="yBZ0Y2t0ceo", help="youtube url id")
    parser.add_argument(
        "--tag", type=int, default=135, help="stream tag, 137=1080p, 136=720p, 135=480p"
    )
    parser.add_argument("--color", type=int, nargs="+", default=[255, 255, 255], help="pose color")
    parser.add_argument("--face", action="store_true", help="plot face keypoints")
    parser.add_argument("--display", action="store_true", help="display inference results")
    parser.add_argument("--fps-size", type=int, default=1)
    parser.add_argument("--gif", action="store_true", help="create gif")
    parser.add_argument("--gif-size", type=int, nargs="+", default=[480, 270])
    parser.add_argument("--kp-size", type=int, default=2, help="keypoint circle size")
    parser.add_argument("--kp-thick", type=int, default=2, help="keypoint circle thickness")
    parser.add_argument("--line-thick", type=int, default=3, help="line thickness")
    parser.add_argument("--alpha", type=float, default=0.3, help="pose alpha")
    parser.add_argument("--kp-obj", action="store_true", help="plot keypoint objects only")

    # model options
    parser.add_argument("--data", type=str, default="data/coco-kp.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--weights", default="kapao_s_coco.pt")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--no-kp-dets", action="store_true", help="do not use keypoint objects")
    parser.add_argument("--conf-thres-kp", type=float, default=0.5)
    parser.add_argument("--conf-thres-kp-person", type=float, default=0.2)
    parser.add_argument("--iou-thres-kp", type=float, default=0.45)
    parser.add_argument("--overwrite-tol", type=int, default=50)
    parser.add_argument("--scales", type=float, nargs="+", default=[1])
    parser.add_argument("--flips", type=int, nargs="+", default=[-1])

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data["imgsz"] = args.imgsz
    data["conf_thres"] = args.conf_thres
    data["iou_thres"] = args.iou_thres
    data["use_kp_dets"] = not args.no_kp_dets
    data["conf_thres_kp"] = args.conf_thres_kp
    data["iou_thres_kp"] = args.iou_thres_kp
    data["conf_thres_kp_person"] = args.conf_thres_kp_person
    data["overwrite_tol"] = args.overwrite_tol
    data["scales"] = args.scales
    data["flips"] = [None if f == -1 else f for f in args.flips]

    device = select_device(args.device, batch_size=1)
    print("Using device: {}".format(device))

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    half = args.half & (device.type != "cpu")
    if half:  # half precision only supported on CUDA
        model.half()
    stride = int(model.stride.max())  # model stride

    imgsz = check_img_size(args.imgsz, s=stride)  # check image size

    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once


    # --------------track stuff------------
    type = "bytetrack"
    tracker = ObjectTracker(type=type)
    conf_thresh = 0.2 if type == "bytetrack" else 0.4
    trail = TrailParser()
    # -------------------------------------
    pause = False
    areas = [[246, 188, 356, 452], [248, 274, 425, 344], [248, 336, 496, 411], [283, 409, 576, 480], [307, 467, 649, 566]]

    write_video = True
    save_dir = "./output"

    if os.path.isdir(args.source):
        test_videos = [osp.join(args.source, s) for s in os.listdir(args.source)]
    else:
        test_videos = [args.source]

    for test_video in test_videos:
        dataset = LoadImages(test_video, img_size=imgsz, stride=stride, auto=True)
        cap = dataset.cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(dataset.frames)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_writer = (
            cv2.VideoWriter(
                osp.join(save_dir, test_video.split(os.sep)[-1]),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps <= 30 else 25,
                (w, h),
            )
            if write_video
            else None
        )

        dataset = tqdm(dataset, desc="Running inference", total=n)
        t0 = time_sync()
        for i, (path, img, im0, _) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            out = model(
                img, augment=True, kp_flip=data["kp_flip"], scales=data["scales"], flips=data["flips"]
            )[0]
            person_dets, kp_dets = run_nms(data, out)
            bboxes, poses, scores, _, _ = post_process_batch(
                data, img, [], [[im0.shape[:2]]], person_dets, kp_dets
            )
            tracks = tracker.update(bboxes=np.array(bboxes), scores=np.array(scores))

            im0_copy = im0.copy()
            for i, track in enumerate(tracks):
                # 行人检测框
                box = [int(b) for b in track[:4]]
                id = track[4]
                pt = ((box[0] + box[2]) // 2, box[3])  # 取脚点
                trail.add_point(id, pt, i)
                trail.plot(id, im0_copy, nums=5, color=(0, 255, 0))

                plot_one_box(
                    box, im0_copy, label=None, color=(0, 255, 0), line_thickness=2
                )
                # cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), (20, 20, 255))
                text = "{}".format(id)
                cv2.putText(
                    im0_copy,
                    text,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            trail.clear_old_points(current_frame=i, interval=100)

            # DRAW POSES
            for j, (bbox, pose) in enumerate(zip(bboxes, poses)):
                x1, y1, x2, y2 = bbox
                # if args.face:
                #     for x, y, c in pose[data['kp_face']]:
                #         if not args.kp_obj or c:
                #             cv2.circle(im0_copy, (int(x), int(y)), args.kp_size, args.color, args.kp_thick)
                # for seg in data['segments'].values():
                #     if not args.kp_obj or (pose[seg[0], -1] and pose[seg[1], -1]):
                #         pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                #         pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                #         cv2.line(im0_copy, pt1, pt2, args.color, args.line_thick)
                seg_wrist = data["segments"][8]
                rw = (int(pose[seg_wrist[1], 0]), int(pose[seg_wrist[1], 1]))
                if point_in_areas(rw, areas):
                    # cv2.circle(im0_copy, (rw[0], rw[1]), args.kp_size, (0, 0, 255), args.kp_thick)
                    cv2.rectangle(
                        im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2
                    )
                # else:
                #     cv2.rectangle(
                #         im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2
                #     )
            im0 = cv2.addWeighted(im0, args.alpha, im0_copy, 1 - args.alpha, gamma=0)

            if write_video:
                vid_writer.write(im0)
            if args.display:
                cv2.namedWindow("p", cv2.WINDOW_NORMAL)
                cv2.imshow("p", im0)
                key = cv2.waitKey(0 if pause else 1)
                pause = True if key == ord(" ") else False
                if key == ord("q") or key == ord("e") or key == 27:
                    exit()

            t1 = time_sync()
            if i == n - 1:
                break

        cap.release()
        if write_video:
            vid_writer.release()
    cv2.destroyAllWindows()
