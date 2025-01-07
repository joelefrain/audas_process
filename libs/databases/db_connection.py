import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        # Cargar variables de entorno desde el archivo .env
        dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/.env'))
        load_dotenv(dotenv_path)
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('DB_PORT')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.database = os.getenv('DB_NAME')

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                auth_plugin='mysql_native_password'
            )
            if self.connection.is_connected():
                print("Conexión exitosa a la base de datos.")
                return self.connection
            else:
                print("No se pudo conectar a la base de datos.")
                return None
        except Error as ex:
            print(f"Error al conectar a la base de datos: {ex}")
            return None

    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            print("Conexión a la base de datos cerrada.")
