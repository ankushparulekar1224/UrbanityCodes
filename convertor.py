from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="c:\\Users\\thero\\Downloads\\Road segmentation.coco-segmentation\\train",
    use_segments=True,
    cls91to80=False
)

convert_coco(
    labels_dir="c:\\Users\\thero\\Downloads\\Road segmentation.coco-segmentation\\valid",
    use_segments=True,
    cls91to80=False
)

convert_coco(
    labels_dir="c:\\Users\\thero\\Downloads\\Road segmentation.coco-segmentation\\test",
    use_segments=True,
    cls91to80=False
)