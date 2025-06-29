"""
Video Processor Module for Meshbuilder
Handles video processing and frame extraction
"""
import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("MeshBuilder.VideoProcessor")

class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config
        
    def extract_frames(self, 
                      video_path: str, 
                      output_dir: Path,
                      frame_rate: float = 1.0) -> List[str]:
        """
        Extract frames from a video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_rate: Number of frames to extract per second
            
        Returns:
            List of paths to extracted frames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_frames = []
        
        logger.info(f"Extracting frames from {video_path} at {frame_rate} fps")
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video {video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            logger.info(f"Video details: {fps} fps, {frame_count} frames, {duration:.2f} seconds")
            
            # Calculate frame interval based on requested frame rate
            frame_interval = max(1, int(fps / frame_rate))
            
            # Extract frames
            frame_idx = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frames at the specified interval
                if frame_idx % frame_interval == 0:
                    # Process for blur detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # Skip blurry frames
                    blur_threshold = self.config.getfloat("Processing", "blur_threshold", fallback=100.0)
                    if laplacian_var < blur_threshold:
                        logger.debug(f"Skipping blurry frame {frame_idx} (var: {laplacian_var:.2f})")
                        frame_idx += 1
                        continue
                    
                    # Save frame
                    frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_frames.append(str(frame_path))
                    
                    saved_count += 1
                    
                    # Log progress periodically
                    if saved_count % 20 == 0:
                        logger.debug(f"Extracted {saved_count} frames so far")
                
                frame_idx += 1
                
            cap.release()
            logger.info(f"Successfully extracted {saved_count} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            
        return extracted_frames
    
    def analyze_video_quality(self, video_path: str) -> Dict[str, float]:
        """
        Analyze video quality
        
        Args:
            video_path: Path to video
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            "blur_score": 0.0,
            "contrast_score": 0.0,
            "stability_score": 0.0,
            "overall_score": 0.0
        }
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video {video_path}")
                return quality_metrics
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for quality assessment
            sample_count = min(30, frame_count)
            sample_interval = max(1, frame_count // sample_count)
            
            blur_scores = []
            contrast_scores = []
            prev_frame = None
            motion_scores = []
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate blur score
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_score = min(1.0, max(0.0, laplacian_var / 1000))
                blur_scores.append(blur_score)
                
                # Calculate contrast score
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist_norm = hist / (gray.shape[0] * gray.shape[1])
                contrast_score = min(1.0, max(0.0, np.std(hist_norm) * 100 / 0.1))
                contrast_scores.append(contrast_score)
                
                # Calculate motion stability if we have a previous frame
                if prev_frame is not None:
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    # Calculate magnitude of flow
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_motion = np.mean(mag)
                    # Convert to stability score (inverse of motion)
                    stability_score = np.exp(-mean_motion / 10)
                    motion_scores.append(stability_score)
                
                prev_frame = gray
            
            # Calculate average scores
            quality_metrics["blur_score"] = np.mean(blur_scores) if blur_scores else 0.0
            quality_metrics["contrast_score"] = np.mean(contrast_scores) if contrast_scores else 0.0
            quality_metrics["stability_score"] = np.mean(motion_scores) if motion_scores else 0.0
            
            # Calculate overall score (weighted average)
            quality_metrics["overall_score"] = (
                0.4 * quality_metrics["blur_score"] +
                0.3 * quality_metrics["contrast_score"] +
                0.3 * quality_metrics["stability_score"]
            )
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error analyzing video quality for {video_path}: {str(e)}")
            
        return quality_metrics