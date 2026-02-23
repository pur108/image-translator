from dataclasses import dataclass

from app.services.ocr import TextRegion


@dataclass
class _Rect:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class TextGrouper:
    """Merge OCR text regions that likely belong to the same speech bubble."""

    def __init__(
        self,
        vertical_gap_ratio: float = 1.5,
        horizontal_overlap_ratio: float = 0.3,
        min_text_length: int = 1,
    ):
        """
        Args:
            vertical_gap_ratio: Max vertical gap between regions as a
                multiple of the average line height. Lines further apart
                than this won't be merged.
            horizontal_overlap_ratio: Minimum horizontal overlap between
                regions (as fraction of the narrower region's width) to
                consider them part of the same column/bubble.
            min_text_length: Regions with text shorter than this are
                filtered out (helps remove watermarks like "FWP").
        """
        self.vertical_gap_ratio = vertical_gap_ratio
        self.horizontal_overlap_ratio = horizontal_overlap_ratio
        self.min_text_length = min_text_length

    def _bbox_to_rect(self, bbox: list[list[int]]) -> _Rect:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return _Rect(min(xs), min(ys), max(xs), max(ys))

    def _rect_to_bbox(self, rect: _Rect) -> list[list[int]]:
        return [
            [rect.x1, rect.y1],
            [rect.x2, rect.y1],
            [rect.x2, rect.y2],
            [rect.x1, rect.y2],
        ]

    def _horizontal_overlap(self, a: _Rect, b: _Rect) -> float:
        """Return horizontal overlap as fraction of the narrower region."""
        overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        narrower = min(a.width, b.width)
        if narrower <= 0:
            return 0.0
        return overlap / narrower

    def _vertical_gap(self, a: _Rect, b: _Rect) -> int:
        """Signed vertical gap: positive = b is below a, negative = overlapping."""
        if a.cy < b.cy:
            return b.y1 - a.y2
        return a.y1 - b.y2

    def _should_merge(self, a: _Rect, b: _Rect, avg_line_height: float) -> bool:
        """Determine if two regions should be merged into the same group."""
        # Must have sufficient horizontal overlap (same column)
        if self._horizontal_overlap(a, b) < self.horizontal_overlap_ratio:
            return False

        # Must be close vertically
        gap = self._vertical_gap(a, b)
        max_gap = avg_line_height * self.vertical_gap_ratio
        return gap <= max_gap

    def _filter_noise(self, regions: list[TextRegion]) -> list[TextRegion]:
        """Remove likely watermarks and noise."""
        filtered: list[TextRegion] = []
        for r in regions:
            text = r.text.strip()
            # Skip very short text (likely watermarks like "FWP")
            if len(text) < self.min_text_length:
                continue
            # Skip single non-alphanumeric characters
            if len(text) == 1 and not text.isalnum():
                continue
            filtered.append(r)
        return filtered

    def group(self, regions: list[TextRegion]) -> list[TextRegion]:
        """
        Group nearby text regions into merged speech bubble regions.

        Returns new TextRegion objects with:
        - bbox: bounding box enclosing all merged regions
        - text: concatenated text (space-separated for English)
        - confidence: minimum confidence of merged regions
        """
        if not regions:
            return []

        regions = self._filter_noise(regions)
        if not regions:
            return []

        # Convert to rects and compute average line height
        rects = [self._bbox_to_rect(r.bbox) for r in regions]
        avg_line_height = sum(r.height for r in rects) / len(rects)

        # Union-Find grouping
        n = len(regions)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Compare all pairs and merge if they belong together
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(rects[i], rects[j], avg_line_height):
                    union(i, j)

        # Build groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Merge each group into a single TextRegion
        merged: list[TextRegion] = []
        for indices in groups.values():
            # Sort by vertical position (top to bottom)
            indices.sort(key=lambda i: rects[i].y1)

            # Merge bounding boxes
            all_x1 = min(rects[i].x1 for i in indices)
            all_y1 = min(rects[i].y1 for i in indices)
            all_x2 = max(rects[i].x2 for i in indices)
            all_y2 = max(rects[i].y2 for i in indices)
            merged_rect = _Rect(all_x1, all_y1, all_x2, all_y2)

            # Concatenate text top-to-bottom
            merged_text = " ".join(regions[i].text.strip() for i in indices)

            # Use minimum confidence
            min_conf = min(regions[i].confidence for i in indices)

            merged.append(
                TextRegion(
                    bbox=self._rect_to_bbox(merged_rect),
                    text=merged_text,
                    confidence=min_conf,
                )
            )

        # Sort final results top-to-bottom, left-to-right (reading order)
        merged.sort(key=lambda r: (r.bbox[0][1], r.bbox[0][0]))
        return merged
