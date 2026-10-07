from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pipeline.feature_extraction.base import BaseFeatureExtractor
from pipeline.feature_extraction.cdch import Cdch
from pipeline.feature_extraction.ldch import Ldch
from pipeline.feature_extraction.rdch import Rdch


@dataclass
class VisualizationMetadata:
    """Metadata for an extractor's visualization."""

    title: str
    x_label: str
    y_label: str
    transpose_features: bool = False


class FeatureExtractorVisualizer:
    """Visualizes geometric feature extraction overlays and resulting heatmaps."""

    def __init__(self, feature_extractors: list[BaseFeatureExtractor]):
        if not feature_extractors:
            raise ValueError("At least one feature extractor must be provided")

        self.feature_extractors = feature_extractors
        self.extractor_handlers: dict[
            type[BaseFeatureExtractor],
            Callable[[BaseFeatureExtractor, np.ndarray], VisualizationMetadata],
        ] = {
            Ldch: self._handle_ldch,
            Rdch: self._handle_rdch,
            Cdch: self._handle_cdch,
        }

    def visualize(self, image: np.ndarray, save_path: str | None = None) -> None:
        """Visualize all extractors side-by-side."""
        num_extractors = len(self.feature_extractors)
        image_height, image_width = image.shape[:2]

        fig, axes = plt.subplots(
            nrows=num_extractors,
            ncols=2,
            figsize=(14, 5 * num_extractors),
            squeeze=False,
        )

        for row_idx, extractor in enumerate(self.feature_extractors):
            ax_img, ax_heat = axes[row_idx]

            # Get visualization metadata for this extractor
            metadata = self._get_visualization_metadata(extractor, image)

            # Extract and prepare features
            features = extractor.extract(image)
            if metadata.transpose_features:
                features = features.T

            # Render without storing state
            self._render_image_overlay(ax_img, image, extractor, metadata)
            self._render_heatmap(ax_heat, features, metadata)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def _get_visualization_metadata(
        self, extractor: BaseFeatureExtractor, image: np.ndarray
    ) -> VisualizationMetadata:
        """Get visualization metadata based on extractor type."""
        handler = self.extractor_handlers.get(type(extractor))

        if handler:
            return handler(extractor, image)
        else:
            return VisualizationMetadata(
                title=type(extractor).__name__,
                x_label="Index",
                y_label="Intensity Bins",
            )

    def _handle_cdch(self, extractor: Cdch, image: np.ndarray) -> VisualizationMetadata:
        """Handle CDCH visualization."""
        abs_center_x, abs_center_y = self._get_center(extractor, image)
        return VisualizationMetadata(
            title=f"CDCH (Px: [{abs_center_x}, {abs_center_y}])",
            x_label="Ray Angle Index",
            y_label="Intensity Bins",
        )

    def _handle_ldch(self, extractor: Ldch, image: np.ndarray) -> VisualizationMetadata:
        """Handle LDCH visualization."""
        return VisualizationMetadata(
            title=f"LDCH ({extractor.sections} {extractor.layout}s)",
            x_label="Intensity Bins",
            y_label="Section Index",
            transpose_features=True,
        )

    def _handle_rdch(self, extractor: Rdch, image: np.ndarray) -> VisualizationMetadata:
        """Handle RDCH visualization."""
        abs_center_x, abs_center_y = self._get_center(extractor, image)
        return VisualizationMetadata(
            title=f"RDCH (Px: [{abs_center_x}, {abs_center_y}])",
            x_label="Ring Index",
            y_label="Intensity Bins",
        )

    def _render_image_overlay(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        extractor: BaseFeatureExtractor,
        metadata: VisualizationMetadata,
    ) -> None:
        """Render image with geometric overlays."""
        image_height, image_width = image.shape[:2]

        ax.imshow(image, cmap="gray")
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_title(metadata.title)
        ax.axis("off")

        # Draw geometry based on extractor type
        if isinstance(extractor, Cdch):
            self._draw_cdch_rays(ax, extractor, image)
        elif isinstance(extractor, Ldch):
            self._draw_ldch_sections(ax, extractor, image)
        elif isinstance(extractor, Rdch):
            self._draw_rdch_rings(ax, extractor, image)

    def _draw_cdch_rays(self, ax: plt.Axes, extractor: Cdch, image: np.ndarray) -> None:
        """Draw CDCH rays from center."""
        abs_center_x, abs_center_y = self._get_center(extractor, image)
        image_height, image_width = image.shape[:2]
        max_radius = int(np.hypot(image_width, image_height))

        all_angles = np.arange(0, 360, extractor.angle_step) - 90
        all_radians = np.deg2rad(all_angles)

        for angle in all_radians:
            end_x = abs_center_x + max_radius * np.cos(angle)
            end_y = abs_center_y + max_radius * np.sin(angle)
            ax.plot(
                [abs_center_x, end_x],
                [abs_center_y, end_y],
                color="red",
                linewidth=0.5,
                alpha=0.7,
            )

        ax.scatter(abs_center_x, abs_center_y, color="lime", s=4, zorder=3)

    def _draw_ldch_sections(
        self, ax: plt.Axes, extractor: Ldch, image: np.ndarray
    ) -> None:
        """Draw LDCH section dividers."""
        image_height, image_width = image.shape[:2]

        if extractor.layout == "row":
            step = image_height / extractor.sections
            for i in range(1, extractor.sections):
                ax.axhline(
                    int(i * step), color="red", linewidth=0.5, alpha=0.7, linestyle="--"
                )
        elif extractor.layout == "col":
            step = image_width / extractor.sections
            for i in range(1, extractor.sections):
                ax.axvline(
                    int(i * step), color="red", linewidth=0.5, alpha=0.7, linestyle="--"
                )

    def _draw_rdch_rings(
        self, ax: plt.Axes, extractor: Rdch, image: np.ndarray
    ) -> None:
        """Draw RDCH concentric rings."""
        abs_center_x, abs_center_y = self._get_center(extractor, image)
        image_height, image_width = image.shape[:2]
        max_radius = self._calculate_max_radius(
            abs_center_x, abs_center_y, image_width, image_height
        )

        num_rings = int(np.ceil(max_radius / extractor.ring_step))

        for r in range(1, num_rings + 1):
            circle = plt.Circle(
                (abs_center_x, abs_center_y),
                r * extractor.ring_step,
                color="red",
                fill=False,
                linewidth=0.5,
                alpha=0.7,
                linestyle="--",
            )
            ax.add_patch(circle)

        ax.scatter(abs_center_x, abs_center_y, color="lime", s=4, zorder=3)

    def _render_heatmap(
        self, ax: plt.Axes, features: np.ndarray, metadata: VisualizationMetadata
    ) -> None:
        """Render heatmap of extracted features."""
        im = ax.imshow(
            features,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )

        ax.set_title("Extracted Features")
        ax.set_xlabel(metadata.x_label)
        ax.set_ylabel(metadata.y_label)

        # Attach colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Normalized Frequency", rotation=270, labelpad=15)

    @staticmethod
    def _get_center(extractor: Cdch | Rdch, image: np.ndarray) -> tuple[int, int]:
        """Get absolute center coordinates."""
        if extractor.center is None:
            extractor.center = (0.5, 0.5)

        image_height, image_width = image.shape[:2]
        abs_center_x = int(extractor.center[0] * image_width)
        abs_center_y = int(extractor.center[1] * image_height)

        return abs_center_x, abs_center_y

    @staticmethod
    def _calculate_max_radius(
        center_x: int, center_y: int, width: int, height: int
    ) -> float:
        """Calculate max radius from center to image corners."""
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
        distances = np.hypot(corners[:, 0] - center_x, corners[:, 1] - center_y)
        return np.max(distances)
