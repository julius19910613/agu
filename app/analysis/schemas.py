from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Size2D(BaseModel):
    width: int
    height: int


class ModelPrediction(BaseModel):
    action_id: int
    action: str
    confidence: float
    probabilities: Dict[str, float]


class MotionFeatures(BaseModel):
    avg_center_speed: float
    max_center_speed: float
    avg_box_area: float
    area_change_ratio: float


class VLMDecisionResponse(BaseModel):
    action: Optional[str]
    confidence: float
    reason: str
    visible_ball: Optional[bool]
    needs_review: bool
    raw_response: str
    available: bool


class FinalDecisionResponse(BaseModel):
    action_id: int
    action: str
    confidence: float
    source: str
    needs_review: bool
    reason: str


class AnalysisRecordResponse(BaseModel):
    player: int
    clip_index: int
    start_frame: int
    end_frame: int
    segment_id: Optional[int] = None
    local_player_id: Optional[str] = None
    global_player_id: Optional[str] = None
    identity_confidence: Optional[float] = None
    r2plus1d: ModelPrediction
    motion: MotionFeatures
    vlm: Optional[VLMDecisionResponse]
    final: FinalDecisionResponse


class AnalysisSummaryResponse(BaseModel):
    clip_count: int
    action_counts: Dict[str, int]
    needs_review_count: int
    source_counts: Dict[str, int]


class VLMVideoAuditResponse(BaseModel):
    available: bool
    player_count_min: Optional[int] = None
    player_count_max: Optional[int] = None
    visible_player_descriptions: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    main_state: str = ""
    confidence: float = 0.0
    limitations: str = ""
    raw_response: str = ""


class LongVideoSegmentResponse(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    player_count: int
    summary: AnalysisSummaryResponse
    vlm_audit: Optional[VLMVideoAuditResponse] = None
    audit_status: str
    audit_notes: List[str] = Field(default_factory=list)


class PlayerBoxScoreEstimateResponse(BaseModel):
    points: int = 0
    assists: int = 0
    rebounds: int = 0
    blocks: int = 0
    steals: int = 0
    confidence: float = 0.0
    method: str = "action_proxy_v1"
    notes: List[str] = Field(default_factory=list)


class LongVideoPlayerSummaryResponse(BaseModel):
    player_id: str
    global_player_id: Optional[str] = None
    identity_confidence: float = 0.0
    identity_method: str = "segment_local"
    identity_evidence: List[str] = Field(default_factory=list)
    segments_seen: int
    clip_count: int
    action_counts: Dict[str, int]
    needs_review_count: int
    average_confidence: float
    statistics: PlayerBoxScoreEstimateResponse = Field(default_factory=PlayerBoxScoreEstimateResponse)


class MergedLongVideoPlayerSummaryResponse(LongVideoPlayerSummaryResponse):
    merged_from_global_player_ids: List[str] = Field(default_factory=list)
    merge_confidence: float = 1.0
    merge_evidence: List[str] = Field(default_factory=list)


class JerseyNumberCandidateResponse(BaseModel):
    number: Optional[str] = None
    confidence: float = 0.0
    visible: bool = False
    source: str = "vlm_jersey_number_v1"
    reason: str = ""
    raw_response: str = ""


class PlayerIdentityFeatureResponse(BaseModel):
    player: int
    segment_id: Optional[int] = None
    local_player_id: Optional[str] = None
    start_frame: int
    end_frame: int
    first_center: List[float] = Field(default_factory=list)
    last_center: List[float] = Field(default_factory=list)
    appearance_signature: Dict[str, float] = Field(default_factory=dict)
    appearance_embedding: List[float] = Field(default_factory=list)
    embedding_model: str = "sidecar_hsv_hist_embedding_v1"
    embedding_dim: int = 0
    track_coverage: float = 0.0
    method: str = "sidecar_hsv_hist_embedding_v1"
    sampled_boxes: List[Dict[str, float]] = Field(default_factory=list)
    jersey_number_candidates: List[JerseyNumberCandidateResponse] = Field(default_factory=list)


class EventCandidateResponse(BaseModel):
    event_type: str
    player_id: Optional[str] = None
    segment_id: Optional[int] = None
    start_frame: int
    end_frame: int
    confidence: float
    method: str
    status: str = "candidate"
    evidence: List[str] = Field(default_factory=list)


class IdentityDuplicateCandidateResponse(BaseModel):
    left_global_player_id: str
    right_global_player_id: str
    confidence: float
    status: str = "candidate_requires_vlm_or_human_confirmation"
    method: str = "identity_duplicate_candidate_v1"
    recommended_action: str = "review_merge"
    left_local_player_ids: List[str] = Field(default_factory=list)
    right_local_player_ids: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    conflict_evidence: List[str] = Field(default_factory=list)


class ConfirmedIdentityMergeResponse(BaseModel):
    canonical_global_player_id: str
    merged_global_player_ids: List[str] = Field(default_factory=list)
    source: str = "manual_review"
    confidence: float = 1.0
    evidence: List[str] = Field(default_factory=list)


class LongVideoAuditSummaryResponse(BaseModel):
    total_segments: int
    passed: int
    warnings: int
    failed: int
    status_counts: Dict[str, int]


class LongVideoAnalysisResponse(BaseModel):
    mode: str = "long_video_segmented"
    duration_sec: float
    fps: float
    frame_count: int
    segment_duration_sec: float
    segment_overlap_sec: float
    segments: List[LongVideoSegmentResponse]
    players: List[LongVideoPlayerSummaryResponse]
    event_candidates: List[EventCandidateResponse] = Field(default_factory=list)
    identity_duplicate_candidates: List[IdentityDuplicateCandidateResponse] = Field(default_factory=list)
    confirmed_identity_merges: List[ConfirmedIdentityMergeResponse] = Field(default_factory=list)
    merged_players: List[MergedLongVideoPlayerSummaryResponse] = Field(default_factory=list)
    audit_summary: LongVideoAuditSummaryResponse


class AnalysisRequest(BaseModel):
    """Payload for running a new analysis pipeline."""
    video_path: str = Field(..., description="Path to the video file to analyze.")
    vlm_mode: str = Field(
        default="low-confidence", 
        description="VLM verification mode: off | low-confidence | always"
    )
    boxes_file: Optional[str] = Field(
        default=None, 
        description="Path to an optional JSON file containing initial bounding boxes."
    )
    max_frames: Optional[int] = Field(
        default=None, 
        description="Optional maximum number of frames to process."
    )
    generate_video: bool = Field(
        default=True,
        description="If True, generates and saves an annotated output video."
    )
    tracker_conf_thres: float = Field(default=0.3, description="YOLO detection confidence threshold (lower = more players).")
    tracker_iou_thres: float = Field(default=0.6, description="YOLO NMS IOU threshold.")
    tracker_min_appear_ratio: float = Field(default=0.02, description="Min ratio of frames a player must appear in (lower = more players).")
    tracker_min_appear_abs: int = Field(default=5, description="Min absolute frame count to keep a player track.")
    vid_stride: Optional[int] = Field(default=None, description="Override default vid_stride (lower = more clips).")
    action_vid_stride: Optional[int] = Field(default=None, description="Override accelerated action clip stride for segmented analysis.")
    tracking_fps: Optional[float] = Field(default=None, description="Optional YOLO tracking frame-rate cap. Lower values speed tracking.")
    yolo_imgsz: Optional[int] = Field(default=None, description="Optional YOLO inference image size. Lower values speed tracking.")
    max_players_per_segment: Optional[int] = Field(default=None, description="Optional cap for active YOLO player tracks per segment.")
    yolo_device: Optional[str] = Field(default=None, description="Optional YOLO device override, for example cpu, cuda, or mps.")
    tracker_backend: Optional[str] = Field(default=None, description="Optional tracker backend override: bytetrack, botsort, or custom.")
    yolo_tracker_config: Optional[str] = Field(default=None, description="Optional Ultralytics tracker YAML, for example bytetrack.yaml or botsort.yaml.")
    yolo_reid_enabled: Optional[bool] = Field(default=None, description="If True, generate a BoT-SORT tracker config with ReID enabled.")
    yolo_reid_model: Optional[str] = Field(default=None, description="ReID model for BoT-SORT, for example auto or a classifier model path.")
    identity_embedding_backend: Optional[str] = Field(default=None, description="Optional identity embedding backend: torchvision_mobilenet_v3_small or sidecar_hsv_hist.")
    identity_embedding_weights: Optional[str] = Field(default=None, description="Optional identity embedding weights: default, imagenet1k_v1, or none.")
    identity_embedding_device: Optional[str] = Field(default=None, description="Optional identity embedding device: auto, cpu, cuda, mps, or mps_if_available.")
    jersey_number_vlm_enabled: Optional[bool] = Field(default=None, description="If True, ask the configured VLM to read jersey numbers from sampled player crops.")
    jersey_number_vlm_frames: Optional[int] = Field(default=None, description="Number of sampled player crops to send for jersey number VLM recognition.")
    confirmed_identity_merges: List[ConfirmedIdentityMergeResponse] = Field(default_factory=list, description="Optional confirmed global-player merges used to produce merged long-video player statistics.")
    r2plus1d_device: Optional[str] = Field(default=None, description="Optional R(2+1)D device override: auto, cpu, cuda, mps, or mps_if_available.")
    low_confidence: Optional[float] = Field(default=None, description="Override default low_confidence threshold.")
    high_confidence: Optional[float] = Field(default=None, description="Override default high_confidence threshold.")
    segmented_analysis: bool = Field(default=True, description="If True, analyze videos through unified overlapped segments.")
    long_video_mode: bool = Field(default=False, description="If True, run segmented long-video analysis with VLM audit.")
    segment_duration_sec: float = Field(default=15.0, description="Long-video segment duration in seconds.")
    segment_overlap_sec: float = Field(default=2.0, description="Overlap between long-video segments in seconds.")
    segment_start_sec: float = Field(default=0.0, description="Start offset for long-video segmented analysis.")
    segment_end_sec: Optional[float] = Field(default=None, description="Optional end offset for long-video segmented analysis.")
    max_segments: Optional[int] = Field(default=None, description="Optional cap on long-video segments for smoke tests.")
    vlm_audit: bool = Field(default=True, description="If True, run VLM contact-sheet audit for each long-video segment.")
    vlm_audit_frames: int = Field(default=6, description="Number of frames to sample into each VLM audit contact sheet.")



class AnalysisResponse(BaseModel):
    """Full payload returned from a successful analysis."""
    video: str
    created_at_unix: float
    runtime_seconds: float
    frame_size: Size2D
    seq_length: int
    vid_stride: int
    tracker_backend: Optional[str] = None
    tracker_config: Optional[str] = None
    reid_enabled: bool = False
    identity_embedding_backend: Optional[str] = None
    identity_embedding_model: Optional[str] = None
    vlm_mode: str
    ollama_model: Optional[str]
    records: list[AnalysisRecordResponse]
    summary: AnalysisSummaryResponse
    player_identity_features: List[PlayerIdentityFeatureResponse] = Field(default_factory=list)
    long_video: Optional[LongVideoAnalysisResponse] = None


class AnalysisRunAsyncResponse(BaseModel):
    task_id: str
    status: str
    message: str


class AnalysisTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    error: Optional[str] = None
    result: Optional[AnalysisResponse] = None
