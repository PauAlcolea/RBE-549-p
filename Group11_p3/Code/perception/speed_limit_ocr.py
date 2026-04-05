"""
perception/speed_limit_ocr.py
=============================
Extract numeric speed limits from detected speed-limit sign crops using EasyOCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import re

import cv2
import numpy as np


@dataclass
class SpeedLimitOcrResult:
    speed_value: Optional[int]
    ocr_confidence: float
    raw_text: str = ""
    has_speed_limit_words: bool = False
    debug_image_path: Optional[str] = None


class SpeedLimitOcr:
    """Lazy-loaded EasyOCR wrapper for speed-limit sign number extraction."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        ocr_cfg = cfg.get("perception", {}).get("speed_limit_ocr") or {}
        self.enabled = bool(ocr_cfg.get("enabled", False))
        self.lang_list = list(ocr_cfg.get("lang_list", ["en"]))
        self.min_confidence = float(ocr_cfg.get("min_confidence", 0.2))
        self.min_speed_mph = int(ocr_cfg.get("min_speed_mph", 5))
        self.max_speed_mph = int(ocr_cfg.get("max_speed_mph", 120))
        self.allowlist = str(ocr_cfg.get("allowlist", "0123456789"))
        self.padding_ratio = float(ocr_cfg.get("padding_ratio", 0.12))
        self.min_crop_size = int(ocr_cfg.get("min_crop_size", 24))
        self.debug_save = bool(ocr_cfg.get("debug_save", False))

        # Use GPU only when requested and available.
        use_gpu = bool(ocr_cfg.get("gpu", True))
        self.use_gpu = bool(use_gpu and device == "cuda")

        configured_cache = ocr_cfg.get("model_storage_directory")
        if configured_cache:
            cache_dir = Path(str(configured_cache))
            if not cache_dir.is_absolute():
                cache_dir = (Path(__file__).resolve().parents[1] / cache_dir).resolve()
        else:
            cache_dir = (Path(__file__).resolve().parents[2] / "Weights" / "easyocr_model_cache").resolve()
        self.model_storage_directory = cache_dir

        configured_user_network_dir = ocr_cfg.get("user_network_directory")
        if configured_user_network_dir:
            user_network_dir = Path(str(configured_user_network_dir))
            if not user_network_dir.is_absolute():
                user_network_dir = (Path(__file__).resolve().parents[1] / user_network_dir).resolve()
        else:
            user_network_dir = (
                Path(__file__).resolve().parents[2] / "Weights" / "easyocr_user_network"
            ).resolve()
        self.user_network_directory = user_network_dir

        self.download_enabled = bool(ocr_cfg.get("download_enabled", True))

        self._reader = None
        self._reader_init_error: Optional[str] = None
        self._reported_init_error = False

    def is_active(self) -> bool:
        return self.enabled

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        if self._reader_init_error is not None:
            raise RuntimeError(self._reader_init_error)

        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed but speed_limit_ocr is enabled. "
                "Install easyocr or disable perception.speed_limit_ocr.enabled."
            ) from exc

        try:
            self.model_storage_directory.mkdir(parents=True, exist_ok=True)
            self.user_network_directory.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self.lang_list,
                gpu=self.use_gpu,
                verbose=True,
                model_storage_directory=str(self.model_storage_directory),
                user_network_directory=str(self.user_network_directory),
                download_enabled=self.download_enabled,
            )
        except Exception as exc:
            self._reader_init_error = (
                "Failed to initialize EasyOCR reader. "
                f"Model cache dir={self.model_storage_directory}. Error: {exc}"
            )
            self.enabled = False
            raise RuntimeError(self._reader_init_error) from exc
        return self._reader

    @staticmethod
    def _sanitize_text(text: str) -> str:
        up = text.upper().replace("O", "0").replace("I", "1").replace("L", "1")
        return up

    @staticmethod
    def _has_speed_limit_words(text: str) -> bool:
        lowered = text.lower()
        letters_only = re.sub(r"[^a-z]", "", lowered)
        has_speed = "speed" in lowered or "speed" in letters_only
        has_limit = "limit" in lowered or "limit" in letters_only
        return has_speed or has_limit

    def _parse_speed(self, text: str) -> Optional[int]:
        sanitized = self._sanitize_text(text)
        matches = re.findall(r"\d{1,3}", sanitized)
        if not matches:
            return None

        values = [int(m) for m in matches]
        for value in values:
            if self.min_speed_mph <= value <= self.max_speed_mph:
                return value
        return None

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _crop(self, frame_bgr: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        if len(bbox) != 4:
            return None

        h, w = frame_bgr.shape[:2]
        x1f, y1f, x2f, y2f = [float(v) for v in bbox]
        bw = max(1.0, x2f - x1f)
        bh = max(1.0, y2f - y1f)
        pad_x = bw * self.padding_ratio
        pad_y = bh * self.padding_ratio

        x1, y1, x2, y2 = [
            int(round(x1f - pad_x)),
            int(round(y1f - pad_y)),
            int(round(x2f + pad_x)),
            int(round(y2f + pad_y)),
        ]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        rh, rw = roi.shape[:2]
        if rh < self.min_crop_size or rw < self.min_crop_size:
            scale = max(self.min_crop_size / max(1, rh), self.min_crop_size / max(1, rw))
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return roi

    @staticmethod
    def _preprocess_variants(roi_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        variants: list[tuple[str, np.ndarray]] = []
        upscaled = cv2.resize(roi_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(("rgb", upscaled))

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )

        variants.append(("gray", gray))
        variants.append(("otsu", otsu))
        variants.append(("otsu_inv", otsu_inv))
        variants.append(("adaptive", adaptive))
        return variants

    @staticmethod
    def _save_debug_panel(
        variants: list[tuple[str, np.ndarray]],
        debug_path: Path,
        title_text: str,
    ) -> None:
        tiles = []
        for name, img in variants[:4]:
            bgr = SpeedLimitOcr._to_bgr(img)
            tile = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.putText(
                tile,
                name,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
            tiles.append(tile)

        while len(tiles) < 4:
            tiles.append(np.zeros((180, 320, 3), dtype=np.uint8))

        top = np.hstack((tiles[0], tiles[1]))
        bottom = np.hstack((tiles[2], tiles[3]))
        panel = np.vstack((top, bottom))

        cv2.putText(
            panel,
            title_text[:120],
            (8, panel.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), panel)

    def infer(
        self,
        frame_bgr: np.ndarray,
        bbox: Sequence[float],
        debug_path: Optional[Path] = None,
    ) -> SpeedLimitOcrResult:
        if not self.enabled:
            return SpeedLimitOcrResult(speed_value=None, ocr_confidence=0.0, has_speed_limit_words=False)

        roi = self._crop(frame_bgr, bbox)
        if roi is None:
            return SpeedLimitOcrResult(speed_value=None, ocr_confidence=0.0, has_speed_limit_words=False)

        variants = self._preprocess_variants(roi)

        try:
            reader = self._get_reader()
        except Exception as exc:
            # Disable OCR after a hard initialization failure to avoid repeated warnings.
            self.enabled = False
            if not self._reported_init_error:
                print(f"[warn] speed-limit OCR initialization failed: {exc}")
                self._reported_init_error = True
            return SpeedLimitOcrResult(speed_value=None, ocr_confidence=0.0, has_speed_limit_words=False)

        best_value = None
        best_conf = 0.0
        best_text = ""
        best_any_text = ""
        best_any_conf = -1.0
        candidate_lines: list[str] = []
        all_text_lines: list[str] = []

        for variant_name, variant_img in variants:
            # Run with strict digit allowlist first; fallback to unrestricted text.
            attempts = [
                {"allowlist": self.allowlist},
                {"allowlist": None},
            ]

            for attempt in attempts:
                kwargs = {
                    "detail": 1,
                    "paragraph": False,
                }
                if attempt["allowlist"] is not None:
                    kwargs["allowlist"] = attempt["allowlist"]

                predictions = reader.readtext(variant_img, **kwargs)
                for pred in predictions:
                    if len(pred) < 3:
                        continue
                    text = str(pred[1])
                    conf = float(pred[2])
                    parsed = self._parse_speed(text)
                    all_text_lines.append(text)
                    if conf > best_any_conf:
                        best_any_conf = conf
                        best_any_text = text
                    candidate_lines.append(
                        f"{variant_name}:{text}:{conf:.2f}:parsed={parsed}"
                    )

                    if conf < self.min_confidence:
                        continue

                    if parsed is None:
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_value = parsed
                        best_text = text

        saved_path = None
        if debug_path is not None and self.debug_save:
            display_text = best_text or best_any_text
            summary = (
                f"best={best_value} conf={best_conf:.2f} text={display_text} "
                f"cands={'; '.join(candidate_lines[:5])}"
            )
            self._save_debug_panel(variants, debug_path, summary)
            saved_path = str(debug_path)

        return_text = best_text or best_any_text
        has_words = self._has_speed_limit_words(" ".join(all_text_lines))
        return SpeedLimitOcrResult(
            speed_value=best_value,
            ocr_confidence=best_conf,
            raw_text=return_text,
            has_speed_limit_words=has_words,
            debug_image_path=saved_path,
        )
