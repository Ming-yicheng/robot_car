"""Voice activity recording with dynamic noise calibration."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from time import time
from typing import Callable, Optional

import numpy as np

from robot_car.config import AudioConfig

logger = logging.getLogger(__name__)


class VoiceRecorder:
    """Record a voice turn to a WAV file using a simple energy threshold."""

    def __init__(self, config: AudioConfig, output_path: Path):
        import pyaudio

        self.pyaudio = pyaudio
        self.config = config
        self.output_path = output_path
        self.dynamic_threshold = float(config.base_threshold)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _format(self):
        return getattr(self.pyaudio, self.config.sample_format_name)

    @staticmethod
    def _volume(data: bytes) -> float:
        return float(np.linalg.norm(np.frombuffer(data, dtype=np.int16)))

    def calibrate_noise(self) -> float:
        """Measure ambient noise once and derive the speaking threshold."""

        audio = self.pyaudio.PyAudio()
        stream = audio.open(
            format=self._format(),
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=self.config.input_device,
            frames_per_buffer=self.config.chunk_size,
        )
        try:
            sample_count = int(self.config.sample_rate / self.config.chunk_size * self.config.calibration_seconds)
            volumes = [
                self._volume(stream.read(self.config.chunk_size, exception_on_overflow=False))
                for _ in range(max(1, sample_count))
            ]
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        noise_floor = float(np.mean(volumes))
        self.dynamic_threshold = noise_floor * 1.2 + 3000.0
        logger.info("Noise floor %.2f, voice threshold %.2f", noise_floor, self.dynamic_threshold)
        return self.dynamic_threshold

    def record_once(self, on_listening: Optional[Callable[[bool], None]] = None) -> Optional[Path]:
        """Record until silence or max duration. Return the WAV path if speech exists."""

        audio = self.pyaudio.PyAudio()
        stream = audio.open(
            format=self._format(),
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=self.config.input_device,
            frames_per_buffer=self.config.chunk_size,
        )
        frames: list[bytes] = []
        recording = False
        silence_started_at: Optional[float] = None
        started_at = time()

        if on_listening:
            on_listening(True)
        logger.info("Listening for voice")

        try:
            while True:
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                volume = self._volume(data)

                if volume > self.dynamic_threshold and not recording:
                    logger.info("Voice detected")
                    recording = True
                    silence_started_at = None

                if recording:
                    frames.append(data)
                    if time() - started_at > self.config.record_seconds:
                        logger.info("Max recording duration reached")
                        break
                    if volume < self.dynamic_threshold:
                        silence_started_at = silence_started_at or time()
                        if (time() - silence_started_at) * 1000 > self.config.silence_duration_ms:
                            logger.info("Silence detected, stopping recording")
                            break
                    else:
                        silence_started_at = None
        finally:
            stream.stop_stream()
            stream.close()
            sample_width = audio.get_sample_size(self._format())
            audio.terminate()
            if on_listening:
                on_listening(False)

        if not frames:
            return None

        with wave.open(str(self.output_path), "wb") as wav:
            wav.setnchannels(self.config.channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(b"".join(frames))
        return self.output_path
