"""项目命令行入口。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="osc-pj")
    commands = parser.add_subparsers(dest="command", required=True)

    road_cmd = commands.add_parser("prepare-road", help="从静态背景提取主干道区域")
    road_cmd.add_argument(
        "--dataset", choices=("XAM-S", "Anthem", "Local", "all"), default="all"
    )

    prelabel_cmd = commands.add_parser("prelabel", help="生成测试帧的车辆框和跨帧 ID")
    prelabel_cmd.add_argument(
        "--dataset", choices=("xam-s", "anthem", "local", "all"), default="all"
    )

    lane_cmd = commands.add_parser("train-lane", help="训练 RoadMark-binary U-Net")
    lane_cmd.add_argument("--data-dir", type=Path, required=True)
    lane_cmd.add_argument("--epochs", type=int)

    video_cmd = commands.add_parser(
        "infer-video", help="处理其他视频并保存检测与跟踪结果"
    )
    video_cmd.add_argument("--video", type=Path, required=True)
    video_cmd.add_argument("--name")
    video_cmd.add_argument("--det-weights", type=Path)
    video_cmd.add_argument("--lane-weights", type=Path)

    commands.add_parser("eval", help="在终端输出检测与跟踪指标")
    commands.add_parser("export-figures", help="从叠加视频导出报告图片")
    args = parser.parse_args()

    if args.command == "prepare-road":
        from preprocess import build_road_mask

        names = (
            ("XAM-S", "Anthem", "Local") if args.dataset == "all" else (args.dataset,)
        )
        for name in names:
            build_road_mask(name)
            print(f"{name} 主干道区域已生成")
    elif args.command == "prelabel":
        from preprocess import prelabel_dataset

        images, boxes = prelabel_dataset(args.dataset)
        print(f"待核对标注：{images} 帧，{boxes} 个车辆框")
    elif args.command == "train-lane":
        from train import train_lane

        print(train_lane(args.data_dir, args.epochs))
    elif args.command == "infer-video":
        from infer import infer_video

        result = infer_video(
            args.video,
            args.name or args.video.stem,
            args.det_weights,
            args.lane_weights,
        )
        print(
            f"视频已生成：{result['processed_frames']} 帧，"
            f"{result['pipeline_fps']:.2f} FPS"
        )
    elif args.command == "export-figures":
        from infer import export_figures

        export_figures()
        print("报告图片已导出")
    elif args.command == "eval":
        from evaluate import evaluate

        metrics = evaluate()
        for name, values in metrics.items():
            print(f"{name} 测试集")
            if name != "平均":
                print(
                    f"  车辆检测：{values['detection_test_images']} 张图，"
                    f"标注框 {values['detection_gt_boxes']}，"
                    f"预测框 {values['detection_pred_boxes']}"
                )
            print(f"  mAP@0.5 {values['map50']:.4f}")
            print(f"  精确率 {values['precision']:.4f}")
            print(f"  召回率 {values['recall']:.4f}")
            print(f"  F1-score {values['f1']:.4f}")
            print(f"  检测 FPS {values['detector_fps']:.2f}")
            if name != "平均":
                print(f"  混淆矩阵 {values['confusion_matrix'].tolist()}")
            if name != "平均":
                print(f"  跟踪：{values['tracking_test_frames']} 帧")
            print(f"  MOTA {values['mota']:.4f}")
            print(f"  ID 跳变次数 {values['idsw']}")
            print(f"  IDF1 {values['idf1']:.4f}")
            print(f"  跟踪 FPS {values['tracking_fps']:.2f}")
            if name == "平均":
                print(f"  ID 跳变总数 {values['idsw_total']}")


if __name__ == "__main__":
    main()
