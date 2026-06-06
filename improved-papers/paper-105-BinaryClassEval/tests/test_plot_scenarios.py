"""Tests covering the actual usage patterns in generate_figures.sh

This test module validates the functionality used when running plot.py
with the various flags and configurations defined in generate_figures.sh.
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from core.curves import plot_net_benefit_curves
from proto.subgroup import SubgroupResults
from reproducibility.seed_manager import get_random_generator


@pytest.fixture
def gender_subgroups():
    """Create Male and Female subgroups simulating eICU demo data."""
    rng = get_random_generator(seed=100, component_name="gender_subgroups")

    # Male subgroup (typically larger in eICU)
    n_male = 150
    y_male = rng.integers(0, 2, size=n_male)
    scores_male = np.where(y_male == 1,
                          rng.uniform(0.5, 0.9, size=n_male),
                          rng.uniform(0.1, 0.6, size=n_male))

    # Female subgroup
    n_female = 120
    y_female = rng.integers(0, 2, size=n_female)
    scores_female = np.where(y_female == 1,
                            rng.uniform(0.5, 0.9, size=n_female),
                            rng.uniform(0.1, 0.6, size=n_female))

    sg_male = SubgroupResults(
        name="Male",
        y_true=y_male,
        y_pred_proba=scores_male,
        display_label="Male"
    )

    sg_female = SubgroupResults(
        name="Female",
        y_true=y_female,
        y_pred_proba=scores_female,
        display_label="Female"
    )

    return [sg_male, sg_female]


@pytest.fixture
def ethnicity_subgroups():
    """Create Caucasian and African American subgroups simulating eICU demo data."""
    rng = get_random_generator(seed=200, component_name="ethnicity_subgroups")

    # Caucasian subgroup (majority)
    n_cauc = 180
    y_cauc = rng.integers(0, 2, size=n_cauc)
    scores_cauc = np.where(y_cauc == 1,
                          rng.uniform(0.5, 0.9, size=n_cauc),
                          rng.uniform(0.1, 0.6, size=n_cauc))

    # African American subgroup (minority)
    n_aa = 80
    y_aa = rng.integers(0, 2, size=n_aa)
    scores_aa = np.where(y_aa == 1,
                        rng.uniform(0.5, 0.9, size=n_aa),
                        rng.uniform(0.1, 0.6, size=n_aa))

    # Add AUC-ROC values (simulated)
    from sklearn.metrics import roc_auc_score
    auc_cauc = roc_auc_score(y_cauc, scores_cauc)
    auc_aa = roc_auc_score(y_aa, scores_aa)

    sg_cauc = SubgroupResults(
        name="Caucasian",
        y_true=y_cauc,
        y_pred_proba=scores_cauc,
        display_label="Caucasian",
        auc_roc=auc_cauc
    )

    sg_aa = SubgroupResults(
        name="African American",
        y_true=y_aa,
        y_pred_proba=scores_aa,
        display_label="African American",
        auc_roc=auc_aa
    )

    return [sg_cauc, sg_aa]


class TestScenario1GenderWithAverages:
    """Test scenario 1 from generate_figures.sh:
    python plot.py --demo --subgroup-field "gender" --subgroups "Male" "Female"
        --minaccuracy 0.75 --style-cycle-offset 8 --average --full-width-average
    """

    def test_gender_subgroups_basic(self, gender_subgroups):
        """Test basic plotting with gender subgroups."""
        ax = plot_net_benefit_curves(
            subgroups=gender_subgroups,
            show_averages=True,
            full_width_average=True,
            min_accuracy=0.75
        )

        assert isinstance(ax, plt.Axes)

        # Check that the plot was created
        assert len(ax.lines) > 0

        # Check y-axis limits include min_accuracy
        ylim = ax.get_ylim()
        assert ylim[0] <= 0.75

        # Close the figure to avoid memory issues
        plt.close(ax.figure)

    def test_style_cycle_offset(self, gender_subgroups):
        """Test that style_cycle_offset shifts colors as expected."""
        # Plot with no offset
        _, ax1 = plt.subplots()
        ax1 = plot_net_benefit_curves(
            subgroups=gender_subgroups,
            ax=ax1,
            style_cycle_offset=None
        )

        # Plot with offset
        _, ax2 = plt.subplots()
        ax2 = plot_net_benefit_curves(
            subgroups=gender_subgroups,
            ax=ax2,
            style_cycle_offset=8
        )

        # Both should create valid plots
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)

        # Get the line colors
        colors1 = [line.get_color() for line in ax1.lines if line.get_linewidth() > 1]
        colors2 = [line.get_color() for line in ax2.lines if line.get_linewidth() > 1]

        # With offset, colors should be different
        if len(colors1) > 0 and len(colors2) > 0:
            assert colors1[0] != colors2[0], "Style offset should change colors"

        plt.close(ax1.figure)
        plt.close(ax2.figure)

    def test_full_width_average(self, gender_subgroups):
        """Test that full_width_average extends lines across full plot."""
        # Without full width
        _, ax1 = plt.subplots()
        ax1 = plot_net_benefit_curves(
            subgroups=gender_subgroups,
            ax=ax1,
            show_averages=True,
            full_width_average=False
        )

        # With full width
        _, ax2 = plt.subplots()
        ax2 = plot_net_benefit_curves(
            subgroups=gender_subgroups,
            ax=ax2,
            show_averages=True,
            full_width_average=True
        )

        # Both should create valid plots
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)

        # Get x-axis limits
        xlim1 = ax1.get_xlim()
        xlim2 = ax2.get_xlim()

        # X-limits should be similar
        assert np.allclose(xlim1, xlim2, atol=0.5)

        plt.close(ax1.figure)
        plt.close(ax2.figure)


class TestScenario2AUCECECalibrationNomain:
    """Test scenario 2 from generate_figures.sh:
    python plot.py --demo --ece --auc --calibration --maxlogodds 0.1 --nomain --minaccuracy 0.8
    """

    def test_auc_visualization(self, ethnicity_subgroups):
        """Test that AUC-ROC horizontal lines are rendered when auc_roc is present."""
        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            hide_main=False,
            max_logodds=np.log(0.1 / (1 - 0.1)),
            min_accuracy=0.8
        )

        assert isinstance(ax, plt.Axes)

        # Check that AUC values are present in the subgroups
        for sg in ethnicity_subgroups:
            assert sg.auc_roc is not None
            assert 0 <= sg.auc_roc <= 1

        # The rendering layer should create horizontal lines for AUC
        # Check that there are enough lines (main curves + AUC lines)
        assert len(ax.lines) >= len(ethnicity_subgroups)

        plt.close(ax.figure)

    def test_calibration_curves(self, ethnicity_subgroups):
        """Test that calibrated curves are computed and rendered."""
        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            compute_calibrated=True,
            max_logodds=np.log(0.1 / (1 - 0.1)),
            min_accuracy=0.8
        )

        assert isinstance(ax, plt.Axes)

        # Check that calibrated curves are rendered
        # Should have more lines than just the main curves
        assert len(ax.lines) >= len(ethnicity_subgroups) * 2

        plt.close(ax.figure)

    def test_hide_main_flag(self, ethnicity_subgroups):
        """Test that hide_main suppresses main curves but keeps other elements."""
        # With main curves
        _, ax1 = plt.subplots()
        ax1 = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            ax=ax1,
            hide_main=False,
            compute_calibrated=True
        )

        # Without main curves
        _, ax2 = plt.subplots()
        ax2 = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            ax=ax2,
            hide_main=True,
            compute_calibrated=True
        )

        # Both should create valid plots
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)

        # Hidden main should have fewer lines
        # (only calibrated curves, not main curves)
        assert len(ax2.lines) < len(ax1.lines)

        plt.close(ax1.figure)
        plt.close(ax2.figure)

    def test_max_logodds_limits(self, ethnicity_subgroups):
        """Test that max_logodds constrains x-axis range."""
        max_logodds_val = np.log(0.1 / (1 - 0.1))

        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            max_logodds=max_logodds_val,
            min_accuracy=0.8
        )

        assert isinstance(ax, plt.Axes)

        # Check x-axis limits
        xlim = ax.get_xlim()
        assert xlim[1] <= max_logodds_val + 0.1  # Allow small tolerance

        # Check y-axis limits
        ylim = ax.get_ylim()
        assert ylim[0] <= 0.8

        plt.close(ax.figure)


class TestScenario3CalibrationOnly:
    """Test scenario 3 from generate_figures.sh:
    python plot.py --demo --calibration --maxlogodds 0.1
    """

    def test_calibration_with_main_curves(self, ethnicity_subgroups):
        """Test calibration with main curves shown (default behavior)."""
        max_logodds_val = np.log(0.1 / (1 - 0.1))

        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            compute_calibrated=True,
            max_logodds=max_logodds_val
        )

        assert isinstance(ax, plt.Axes)

        # Should have both main and calibrated curves
        # Each subgroup should contribute at least 2 lines (main + calibrated)
        assert len(ax.lines) >= len(ethnicity_subgroups) * 2

        # Check legend is present
        legend = ax.get_legend()
        assert legend is not None

        plt.close(ax.figure)

    def test_calibration_changes_curves(self, ethnicity_subgroups):
        """Verify that calibrated curves exist and are computed correctly."""
        from core.computation import prepare_data_for_visualization
        from proto.config import ComputationConfig
        from stats.prevalence import default_prevalence_grid

        prevalence_grid = default_prevalence_grid(num=50)

        # Config with calibration
        config = ComputationConfig(
            prevalence_grid=prevalence_grid,
            cost_ratio=1.0,
            compute_calibrated=True,
            normalize=True
        )

        computed = prepare_data_for_visualization(ethnicity_subgroups, config)

        # Check that calibrated curves exist
        for data in computed:
            assert data.calibrated_nb_curve is not None
            assert len(data.calibrated_nb_curve) == len(data.nb_curve)
            assert not np.isnan(data.calibrated_nb_curve).any()
            # Calibrated curves should be valid probabilities (with normalization)
            assert ((0 <= data.calibrated_nb_curve) & (data.calibrated_nb_curve <= 1)).all()


class TestScenario4AveragesOnly:
    """Test scenario 4 from generate_figures.sh:
    python plot.py --demo --average --maxlogodds 0.1
    """

    def test_averages_with_maxlogodds(self, ethnicity_subgroups):
        """Test average lines with constrained x-axis."""
        max_logodds_val = np.log(0.1 / (1 - 0.1))

        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            show_averages=True,
            max_logodds=max_logodds_val
        )

        assert isinstance(ax, plt.Axes)

        # Should have main curves + average lines
        assert len(ax.lines) >= len(ethnicity_subgroups) * 2

        # Check x-axis limits
        xlim = ax.get_xlim()
        assert xlim[1] <= max_logodds_val + 0.1

        plt.close(ax.figure)

    def test_average_lines_properties(self, ethnicity_subgroups):
        """Test that average lines have expected visual properties."""
        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            show_averages=True,
            full_width_average=False
        )

        assert isinstance(ax, plt.Axes)

        # Average lines should be thinner/more transparent
        # Check that some lines have lower alpha or different width
        alphas = [line.get_alpha() for line in ax.lines if line.get_alpha() is not None]
        linewidths = [line.get_linewidth() for line in ax.lines]

        # Should have variety in alphas or linewidths
        assert len(set(alphas)) > 1 or len(set(linewidths)) > 1

        plt.close(ax.figure)


class TestIntegrationAllScenarios:
    """Integration tests covering combinations of features from all scenarios."""

    def test_all_features_combined(self, ethnicity_subgroups):
        """Test using multiple features together."""
        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            compute_calibrated=True,
            compute_ci=True,
            n_bootstrap=20,  # Small number for speed
            show_averages=True,
            full_width_average=True,
            show_diamonds=True,
            diamond_shift_amount=0.3,
            max_logodds=np.log(0.5 / 0.5),
            min_accuracy=0.75,
            ci_alpha=0.2,
            random_seed=42
        )

        assert isinstance(ax, plt.Axes)

        # Should have multiple visual elements
        assert len(ax.lines) > 0
        assert len(ax.collections) > 0  # CIs and/or scatter points

        # Check for confidence intervals (PolyCollection)
        ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(ci_polys) > 0, "Should have CI polygons"

        plt.close(ax.figure)

    def test_reproducibility_with_seed(self, ethnicity_subgroups):
        """Test that random_seed ensures reproducible results."""
        # First run
        ax1 = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            compute_ci=True,
            n_bootstrap=20,
            random_seed=123
        )

        # Get the lines data
        lines1_data = [(line.get_xdata().copy(), line.get_ydata().copy())
                      for line in ax1.lines]

        plt.close(ax1.figure)

        # Second run with same seed
        ax2 = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            compute_ci=True,
            n_bootstrap=20,
            random_seed=123
        )

        lines2_data = [(line.get_xdata().copy(), line.get_ydata().copy())
                      for line in ax2.lines]

        # Should have same number of lines
        assert len(lines1_data) == len(lines2_data)

        # Lines should be identical
        for (x1, y1), (x2, y2) in zip(lines1_data, lines2_data):
            assert np.allclose(x1, x2), "X data should be identical"
            assert np.allclose(y1, y2), "Y data should be identical"

        plt.close(ax2.figure)

    def test_empty_subgroups_handling(self):
        """Test handling of edge case with empty or minimal subgroups."""
        # Very small subgroup (edge case)
        rng = get_random_generator(seed=999, component_name="small_subgroup")
        y_small = np.array([0, 1, 0, 1, 1])
        scores_small = rng.uniform(0, 1, size=5)

        sg_small = SubgroupResults(
            name="Small",
            y_true=y_small,
            y_pred_proba=scores_small
        )

        # Should not crash
        ax = plot_net_benefit_curves(
            subgroups=[sg_small],
            compute_ci=False  # Skip CI for tiny dataset
        )

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) > 0

        plt.close(ax.figure)


class TestSpecificPlotFeatures:
    """Tests for specific plot features used in generate_figures.sh."""

    def test_legend_customization(self, ethnicity_subgroups):
        """Test custom legend mapping."""
        custom_mapping = {
            "Caucasian": "Group A",
            "African American": "Group B"
        }

        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            subgroup_legend_mapping=custom_mapping
        )

        assert isinstance(ax, plt.Axes)

        # Check legend exists
        legend = ax.get_legend()
        assert legend is not None

        # Check legend text includes custom labels
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("Group A" in text for text in legend_texts)
        assert any("Group B" in text for text in legend_texts)

        plt.close(ax.figure)

    def test_diamonds_with_shift(self, ethnicity_subgroups):
        """Test diamond markers with shift amount."""
        ax = plot_net_benefit_curves(
            subgroups=ethnicity_subgroups,
            show_diamonds=True,
            diamond_shift_amount=0.5
        )

        assert isinstance(ax, plt.Axes)

        # Should have scatter collections (for diamonds and training points)
        scatter_collections = [c for c in ax.collections
                              if not isinstance(c, PolyCollection)]
        assert len(scatter_collections) > 0

        plt.close(ax.figure)

    def test_varying_cost_ratios(self, ethnicity_subgroups):
        """Test different cost_ratio values (FP cost / FN cost)."""
        cost_ratios = [0.25, 0.5, 0.75, 1.0]

        for cost_ratio in cost_ratios:
            ax = plot_net_benefit_curves(
                subgroups=ethnicity_subgroups,
                cost_ratio=cost_ratio
            )

            assert isinstance(ax, plt.Axes)
            assert len(ax.lines) > 0

            plt.close(ax.figure)

    def test_normalize_flag_effects(self, ethnicity_subgroups):
        """Test effect of normalization on net benefit values."""
        from core.computation import prepare_data_for_visualization
        from proto.config import ComputationConfig
        from stats.prevalence import default_prevalence_grid

        grid = default_prevalence_grid(num=30)

        # With normalization
        config_norm = ComputationConfig(
            prevalence_grid=grid,
            cost_ratio=0.5,
            normalize=True
        )

        # Without normalization
        config_no_norm = ComputationConfig(
            prevalence_grid=grid,
            cost_ratio=0.5,
            normalize=False
        )

        computed_norm = prepare_data_for_visualization(ethnicity_subgroups, config_norm)
        computed_no_norm = prepare_data_for_visualization(ethnicity_subgroups, config_no_norm)

        # With normalization, values should be in [0, 1]
        for data in computed_norm:
            assert ((0 <= data.nb_curve) & (data.nb_curve <= 1)).all()

        # Without normalization, values may exceed 1
        # Just check they're computed and valid (no NaN)
        for data in computed_no_norm:
            assert not np.isnan(data.nb_curve).any()


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""

    def test_single_subgroup(self, gender_subgroups):
        """Test plotting with a single subgroup."""
        ax = plot_net_benefit_curves(
            subgroups=[gender_subgroups[0]],  # Just Male
            show_averages=False  # Averages require 2+ subgroups
        )

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) > 0

        plt.close(ax.figure)

    def test_perfect_separation(self):
        """Test with perfectly separated data."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        sg = SubgroupResults(
            name="Perfect",
            y_true=y_true,
            y_pred_proba=scores
        )

        ax = plot_net_benefit_curves(
            subgroups=[sg],
            compute_ci=False
        )

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_no_separation(self):
        """Test with completely random (no separation) data."""
        rng = get_random_generator(seed=777, component_name="random_classifier")
        y_true = rng.integers(0, 2, size=100)
        scores = rng.uniform(0, 1, size=100)

        sg = SubgroupResults(
            name="Random",
            y_true=y_true,
            y_pred_proba=scores
        )

        ax = plot_net_benefit_curves(
            subgroups=[sg],
            compute_ci=False
        )

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)
