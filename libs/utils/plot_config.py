import matplotlib.pyplot as plt
import locale

class PlotConfig:
    """Class to configure global matplotlib parameters."""

    # Global matplotlib variables
    font_family = "Arial"
    legend_loc = "upper right"

    @classmethod
    def setup_matplotlib(cls):
        """Configure global matplotlib parameters."""

        # Set font family and legend location
        plt.rcParams["font.family"] = cls.font_family
        plt.rcParams["legend.loc"] = cls.legend_loc

        # Enable locale settings for number formatting
        plt.rcParams["axes.formatter.use_locale"] = True

        # Set locale to use comma as decimal separator
        try:
            locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")
        except locale.Error:
            print(
                "The locale 'es_ES.UTF-8' is not available. Using default settings."
            )

        # Automatically configure tight_layout for all figures
        plt.rcParams["figure.constrained_layout.h_pad"] = 0
        plt.rcParams["figure.constrained_layout.hspace"] = 0
        plt.rcParams["figure.constrained_layout.w_pad"] = 0
        plt.rcParams["figure.constrained_layout.wspace"] = 0
        plt.rcParams["figure.constrained_layout.use"] = True

        # Set matplotlib backend
        plt.rcParams["backend"] = 'Agg'

        # Configure date format
        plt.rcParams["date.autoformatter.day"] = '%d-%m-%Y'
        plt.rcParams["date.autoformatter.day"] = '%H-%M-%S'

        # Set figure edge and face colors
        plt.rcParams["figure.edgecolor"] = 'None'
        plt.rcParams["figure.facecolor"] = 'None'