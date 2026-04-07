import os
import torch
import yaml
from ultralytics import YOLO


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def build_dataset_paths():
    project_root = get_project_root()
    results_root = os.path.join(project_root, "SMVL_logo_seg_results_sam3b_yolo")

    # Với run_sam hiện tại, DATASET_NAME = "SMVL_JPEGImages"
    ds_name = "SMVL_JPEGImages"
    output_dir = os.path.join(results_root, ds_name)
    yolo_ds_root = os.path.join(output_dir, "yolo_dataset")

    images_train = os.path.join(yolo_ds_root, "images", "train")
    images_val = os.path.join(yolo_ds_root, "images", "val")
    labels_train = os.path.join(yolo_ds_root, "labels", "train")
    labels_val = os.path.join(yolo_ds_root, "labels", "val")
    classes_path = os.path.join(output_dir, "classes.txt")

    return {
        "output_dir": output_dir,
        "yolo_ds_root": yolo_ds_root,
        "images_train": images_train,
        "images_val": images_val,
        "labels_train": labels_train,
        "labels_val": labels_val,
        "classes_path": classes_path,
    }


def load_class_names(classes_path):
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Khong tim thay classes.txt tai: {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    if not names:
        raise ValueError("classes.txt rong, khong co ten class nao.")
    return names


def write_data_yaml(ds_paths, names):
    data_yaml_path = os.path.join(ds_paths["yolo_ds_root"], "smvl_seg.yaml")
    os.makedirs(ds_paths["yolo_ds_root"], exist_ok=True)

    data = {
        "path": ds_paths["yolo_ds_root"],
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    return data_yaml_path


def main():
    ds_paths = build_dataset_paths()

    # Kiểm tra dataset đã sẵn sàng chưa
    for p in [
        ds_paths["images_train"],
        ds_paths["images_val"],
        ds_paths["labels_train"],
        ds_paths["labels_val"],
    ]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Thu muc khong ton tai: {p}\nHay dam bao da chay run_sam.py truoc.")

    names = load_class_names(ds_paths["classes_path"])
    data_yaml_path = write_data_yaml(ds_paths, names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Su dung device: {device}")
    print(f"Data YAML: {data_yaml_path}")

    # Model seg nho, phu hop realtime / GPU vua phai
    model_name = "yolo11s-seg.pt"
    model = YOLO(model_name)

    # Cấu hình train mặc định tương đối nhẹ, có thể chỉnh thêm nếu cần
    model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        device=0 if device == "cuda" else "cpu",
        batch=-1,  # auto batch theo GPU
        workers=4,
        project=os.path.join(get_project_root(), "yolo_seg_runs"),
        name="smvl_yolo11s_seg",
        verbose=True,
    )


if __name__ == "__main__":
    main()

