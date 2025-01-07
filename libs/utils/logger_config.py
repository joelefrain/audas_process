import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def get_logger(module_name):
    # Crear un logger para el módulo específico
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Obtener la fecha actual para crear la carpeta correspondiente
    fecha_actual = datetime.now().strftime('%Y-%m-%d')

    # Crear un directorio de logs si no existe, agrupado por fecha
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs', fecha_actual))
    os.makedirs(log_dir, exist_ok=True)

    # Configurar el archivo de log para este módulo dentro de la carpeta de la fecha actual
    log_file = os.path.join(log_dir, f'{module_name}.log')

    # Crear un formateador
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Crear un handler de rotación de logs
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Crear un handler para la consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Agregar los handlers al logger si no existen
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger