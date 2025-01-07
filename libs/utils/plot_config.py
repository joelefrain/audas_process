import matplotlib.pyplot as plt
import matplotlib
import locale
import os


class PlotConfig:
    """Clase para configurar parámetros globales de matplotlib."""

    # Variables globales de matplotlib
    font_family = "Arial"
    legend_loc = "upper right"

    @classmethod
    def setup_matplotlib(cls):
        """Configura los parámetros globales de matplotlib."""
        if os.name == "posix":
            # En Linux, usa una fuente alternativa si 'Arial' no está disponible
            cls.font_family = "DejaVu Sans"

        plt.rcParams["font.family"] = cls.font_family
        plt.rcParams["legend.loc"] = cls.legend_loc
        plt.rcParams["axes.formatter.use_locale"] = True

        # Establece la configuración regional para usar coma como separador decimal
        try:
            locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")
        except locale.Error:
            print(
                "La configuración regional 'es_ES.UTF-8' no está disponible. Usando configuración predeterminada."
            )

        # Establece un backend no interactivo para evitar que los gráficos se muestren
        matplotlib.use("Agg")
