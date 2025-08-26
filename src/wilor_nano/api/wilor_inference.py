import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import rerun as rr
import torch
from jaxtyping import Int
from numpy import ndarray
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import VideoReader

from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


@dataclass
class WilorConfig:
    rr_config: RerunTyroConfig
    image_path: Path | None = None
    video_path: Path | None = None


def set_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
            ]
        ),
        static=True,
    )


def main(config: WilorConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    set_annotation_context()

    # make sure one is not none
    assert config.image_path is not None or config.video_path is not None
    if config.image_path:
        image = cv2.imread(str(config.image_path))
        outputs: list[dict] = pipe.predict(image)
        rr.log("image", rr.Image(image, color_model=rr.ColorModel.BGR))

        for output in outputs:
            handedness: Literal["left", "right"] = "right" if output["is_right"] == 1.0 else "left"
            hand_bbox = output["hand_bbox"]
            hand_keypoints = output["wilor_preds"]["pred_keypoints_2d"]
            print(output)
            xyz = output["wilor_preds"]["pred_keypoints_3d"]
            rr.log(f"{handedness}_xyz", rr.Points3D(positions=xyz))
            rr.log(
                f"image/{handedness}_box",
                rr.Boxes2D(array=hand_bbox, array_format=rr.Box2DFormat.XYXY, show_labels=True),
            )
            rr.log(f"image/{handedness}_keypoints", rr.Points2D(positions=hand_keypoints))

    if config.video_path:
        video_reader = VideoReader(filename=config.video_path)
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            video_path=config.video_path, video_log_path=Path("video"), timeline="video_time"
        )
        for ts, bgr in zip(frame_timestamps_ns, video_reader, strict=False):
            rr.set_time("video_time", duration=1e-9 * ts)
            outputs: list[dict] = pipe.predict(bgr)
            for output in outputs:
                handedness: Literal["left", "right"] = "right" if output["is_right"] == 1.0 else "left"
                hand_bbox = output["hand_bbox"]
                hand_keypoints = output["wilor_preds"]["pred_keypoints_2d"]
                xyz = output["wilor_preds"]["pred_keypoints_3d"]
                rr.log(
                    f"{handedness}_xyz",
                    rr.Points3D(
                        positions=xyz,
                        class_ids=0,
                        keypoint_ids=MEDIAPIPE_IDS,
                        show_labels=False,
                        colors=(0, 255, 0),
                    ),
                )

                rr.log(
                    f"video/{handedness}_box",
                    rr.Boxes2D(array=hand_bbox, array_format=rr.Box2DFormat.XYXY, show_labels=True),
                )
                rr.log(
                    f"video/{handedness}_keypoints",
                    rr.Points2D(
                        positions=hand_keypoints,
                        class_ids=0,
                        keypoint_ids=MEDIAPIPE_IDS,
                        show_labels=False,
                        colors=(0, 255, 0),
                    ),
                )

    # print(outputs)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # hand_bboxs = []
    # is_rights = []
    # for i in range(len(outputs)):
    #     hand_bboxs.append(outputs[i]["hand_bbox"])
    #     is_rights.append(outputs[i]["is_right"])
    # for _ in range(100):
    #     t0 = time.time()
    #     outputs = pipe.predict_with_bboxes(image, np.array(hand_bboxs), is_rights)
    #     print(time.time() - t0)
