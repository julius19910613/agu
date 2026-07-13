from __future__ import annotations

import numpy as np

from app.analysis.scoreboard_ocr import RapidOCRScoreboardReader


class FakeOCR:
    def __init__(self, results):
        self.results = results

    def __call__(self, image, **kwargs):
        return self.results, [0.0, 0.0, 0.0]


def _reader(results):
    reader = object.__new__(RapidOCRScoreboardReader)
    reader._ocr = FakeOCR(results)
    reader.confidence_threshold = 0.75
    return reader


def test_rapidocr_reader_selects_large_separated_side_scores():
    reader = _reader(
        [
            [[[190, 100], [325, 100], [325, 190], [190, 190]], "38", 0.96],
            [[[390, 90], [640, 90], [640, 145], [390, 145]], "00:45.3", 0.89],
            [[[700, 98], [835, 98], [835, 190], [700, 190]], "36", 0.99],
            [[[480, 40], [520, 40], [520, 70], [480, 70]], "1", 0.92],
        ]
    )

    image = np.zeros((420, 1200, 3), dtype=np.uint8)
    image[100:191, 190:326] = (255, 255, 0)
    image[98:191, 700:836] = (255, 255, 0)
    result = reader.read(image)

    assert result is not None
    assert (result.left_score, result.right_score) == (38, 36)
    assert result.confidence == 0.96


def test_rapidocr_reader_rejects_unpaired_or_low_confidence_digits():
    reader = _reader(
        [
            [[[190, 100], [325, 100], [325, 190], [190, 190]], "38", 0.96],
            [[[700, 98], [835, 98], [835, 190], [700, 190]], "36", 0.40],
        ]
    )

    image = np.zeros((420, 1200, 3), dtype=np.uint8)
    image[100:191, 190:326] = (255, 255, 0)
    image[98:191, 700:836] = (255, 255, 0)
    assert reader.read(image) is None


def test_rapidocr_reader_rejects_likely_truncated_late_game_score():
    reader = _reader(
        [
            [[[190, 100], [325, 100], [325, 190], [190, 190]], "99", 0.99],
            [[[700, 98], [770, 98], [770, 190], [700, 190]], "0", 0.99],
        ]
    )
    image = np.zeros((420, 1200, 3), dtype=np.uint8)
    image[100:191, 190:326] = (255, 255, 0)
    image[98:191, 700:771] = (255, 255, 0)

    assert reader.read(image) is None


def test_rapidocr_reader_disables_orientation_for_seven_segment_scores():
    class OrientationFakeOCR:
        def __init__(self):
            self.recognition_calls = 0

        def __call__(self, image, **kwargs):
            if kwargs.get("use_det") is False:
                values = [("21", 0.97), ("19", 0.93)]
                value = values[self.recognition_calls]
                self.recognition_calls += 1
                return [[value[0], value[1]]], [0.0]
            return [
                [[[190, 100], [325, 100], [325, 190], [190, 190]], "21", 0.99],
                [[[700, 98], [835, 98], [835, 190], [700, 190]], "61", 0.98],
            ], [0.0, 0.0, 0.0]

    reader = object.__new__(RapidOCRScoreboardReader)
    reader._ocr = OrientationFakeOCR()
    reader.confidence_threshold = 0.75
    image = np.zeros((420, 1200, 3), dtype=np.uint8)
    image[100:191, 190:326] = (255, 255, 0)
    image[98:191, 700:836] = (255, 255, 0)

    result = reader.read(image)

    assert result is not None
    assert (result.left_score, result.right_score) == (21, 19)
    assert result.confidence == 0.93
