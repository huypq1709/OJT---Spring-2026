"""
Script detect objects trong video bằng YOLOv11.

Usage:
    python detect_video.py --video path/to/video.mp4
    python detect_video.py --video path/to/video.mp4 --conf 0.5 --save-frames
"""

import argparse
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json


class VideoDetector:
    """Detect objects trong video bằng YOLOv11."""
    
    def __init__(self, model_path="model/yolov11l_model.pt"):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    
    def detect_video(
        self,
        video_path: str,
        output_dir: str = "output/detections",
        conf_threshold: float = 0.5,
        sample_rate: int = 1,
        save_frames: bool = False,
        save_crops: bool = False,
        save_video: bool = False
    ):
        """
        Detect objects trong video.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory
            conf_threshold: Confidence threshold (0.0-1.0)
            sample_rate: Process every N frames
            save_frames: Save frames with bounding boxes
            save_crops: Save cropped detections
            save_video: Save output video with bounding boxes
            
        Returns:
            Dictionary with detection results
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        if save_frames:
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
        
        if save_crops:
            crops_dir = os.path.join(output_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
        
        # Video writer for output
        video_writer = None
        if save_video:
            output_video_path = os.path.join(output_dir, "output_video.mp4")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Video Info:")
        print(f"   Path: {video_path}")
        print(f"   Frames: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Sample rate: Every {sample_rate} frame(s)")
        print(f"   Confidence threshold: {conf_threshold}\n")
        
        # Initialize video writer if needed
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Detection results
        all_detections = []
        detection_summary = {}
        frame_idx = 0
        processed_frames = 0
        total_detections = 0
        
        with tqdm(total=total_frames, desc="Detecting") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                if frame_idx % sample_rate == 0:
                    # Run detection
                    results = self.model(frame, conf=conf_threshold, verbose=False)
                    
                    frame_detections = []
                    
                    for result in results:
                        boxes = result.boxes
                        
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
                            class_name = self.model.names[cls] if cls < len(self.model.names) else f"class_{cls}"
                            
                            detection = {
                                'frame_idx': frame_idx,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': cls,
                                'class_name': class_name
                            }
                            
                            frame_detections.append(detection)
                            
                            # Update summary
                            if class_name not in detection_summary:
                                detection_summary[class_name] = 0
                            detection_summary[class_name] += 1
                            
                            # Draw bounding box
                            if save_frames or save_crops or save_video:
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw on frame
                                if save_frames or save_video:
                                    color = (0, 255, 0)  # Green
                                    thickness = 2
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                                    
                                    # Label - ONLY CONFIDENCE (no class name)
                                    label = f"{conf:.2f}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.6
                                    font_thickness = 2
                                    
                                    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                                    # Draw background for label
                                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
                                    # Draw text
                                    cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
                                
                                # Save crop
                                if save_crops:
                                    crop = frame[y1:y2, x1:x2]
                                    crop_filename = f"frame_{frame_idx:06d}_det_{idx}_{class_name}.jpg"
                                    crop_path = os.path.join(crops_dir, crop_filename)
                                    cv2.imwrite(crop_path, crop)
                    
                    # Save frame with boxes
                    if save_frames and len(frame_detections) > 0:
                        frame_filename = f"frame_{frame_idx:06d}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                    
                    if frame_detections:
                        all_detections.extend(frame_detections)
                        total_detections += len(frame_detections)
                    
                    processed_frames += 1
                
                # Write frame to video (nếu có bboxes hoặc không có cũng ghi)
                if save_video:
                    video_writer.write(frame)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Release video writer
        if video_writer is not None:
            video_writer.release()
            print(f"\n✅ Video output saved!")
        
        # Save results to JSON
        results_data = {
            'video_path': video_path,
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'resolution': [width, height]
            },
            'detection_config': {
                'conf_threshold': conf_threshold,
                'sample_rate': sample_rate,
                'processed_frames': processed_frames
            },
            'summary': {
                'total_detections': total_detections,
                'detections_by_class': detection_summary
            },
            'detections': all_detections
        }
        
        results_path = os.path.join(output_dir, "detections.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print("📊 DETECTION SUMMARY")
        print('='*70)
        print(f"Processed frames: {processed_frames}/{total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"\nDetections by class:")
        for class_name, count in sorted(detection_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
        
        print(f"\n💾 Results saved to:")
        print(f"  JSON: {results_path}")
        if save_video:
            print(f"  Video: {output_video_path}")
        if save_frames:
            print(f"  Frames: {frames_dir}")
        if save_crops:
            print(f"  Crops: {crops_dir}")
        print('='*70)
        
        return results_data


def main():
    parser = argparse.ArgumentParser(description="Detect objects in video using YOLOv11")
    
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file")
    parser.add_argument("--model", type=str, default="model/yolov11s_model.pt",
                        help="Path to YOLO model (default: model/yolov11s_model.pt)")
    parser.add_argument("--output", type=str, default="output/detections",
                        help="Output directory (default: output/detections)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold 0.0-1.0 (default: 0.5)")
    parser.add_argument("--sample-rate", type=int, default=1,
                        help="Process every N frames (default: 1 = all frames)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save output video with bounding boxes (confidence only)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save frames with bounding boxes")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save cropped detections")
    
    args = parser.parse_args()
    
    # Check video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video not found: {args.video}")
        return
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        return
    
    # Initialize detector
    detector = VideoDetector(model_path=args.model)
    
    # Run detection
    detector.detect_video(
        video_path=args.video,
        output_dir=args.output,
        conf_threshold=args.conf,
        sample_rate=args.sample_rate,
        save_frames=args.save_frames,
        save_crops=args.save_crops,
        save_video=args.save_video
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
