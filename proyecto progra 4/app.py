from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pandas.io.sql as sql
import psycopg2
import matplotlib
matplotlib.use('Agg') # Importante para evitar errores de GUI en el servidor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from contextlib import closing
import io
import base64

# --- INICIALIZACIÓN DE FLASK (Corregido) ---
app = Flask(__name__) # Se usan dos guiones bajos a cada lado
CORS(app)  # Habilita CORS para permitir llamadas desde el frontend

# --- CONFIGURACIÓN DE POSTGRESQL ---
# ¡OJO! Cambia la contraseña si la tuya no es 'root'
DB_CONFIG = {
    'dbname': 'proyecto_ml',
    'user': 'postgres',
    'password': 'root', 
    'port': '5432',
    'host': 'localhost'
}

# --- LÓGICA DEL MODELO ---
modelo = LinearRegression()
mae = 0.0
rmse = 0.0
X = pd.DataFrame() # Inicializar vacío para evitar errores
y = pd.Series(dtype=float)

# Función para entrenar el modelo con los datos de la base de datos
def entrenar_modelo():
    global modelo, mae, rmse, X, y
    try:
        print("Intentando conectar a la base de datos y entrenar...")
        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            # Asegúrate de que los nombres de columnas coincidan con tu SQL
            query = "SELECT asistencia_porcentaje, tareas_completadas, participacion, calificacion_final FROM calificaciones;"
            df = sql.read_sql(query, conn)

        if df.empty:
            print("AVISO: No hay datos en la BD. Usando datos simulados para iniciar.")
            # Datos simulados de respaldo
            data = {
                'asistencia_porcentaje': np.random.randint(60, 100, 100),
                'tareas_completadas': np.random.randint(0, 10, 100),
                'participacion': np.random.randint(0, 10, 100),
                'calificacion_final': np.random.randint(60, 100, 100)
            }
            df = pd.DataFrame(data)

        # Variables independientes (X) y dependientes (y)
        # Usamos los nombres exactos de la base de datos
        X = df[['asistencia_porcentaje', 'tareas_completadas', 'participacion']]
        y = df['calificacion_final']

        # Entrenar el modelo
        modelo.fit(X, y)

        # Evaluar el modelo
        predicciones = modelo.predict(X)
        mae = mean_absolute_error(y, predicciones)
        rmse = np.sqrt(mean_squared_error(y, predicciones))

        print("Modelo entrenado exitosamente.")
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    except Exception as e:
        print(f"FALLO CRÍTICO al entrenar el modelo: {e}")
        print("Verifica que la base de datos 'proyecto_ml' y la tabla 'calificaciones' existan.")

# --- ENDPOINT PARA GUARDAR LOS DATOS (Opcional, si quieres guardar predicciones) ---
@app.route('/api/guardar', methods=['POST'])
def guardar_datos():
    try:
        datos = request.get_json()
        asistencia = float(datos['asistencia'])
        tareas = float(datos['tareas'])
        participacion = float(datos['participacion'])
        calificacion_final = float(datos['calificacion_predicha'])

        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO calificaciones (asistencia_porcentaje, tareas_completadas, participacion, calificacion_final)
                VALUES (%s, %s, %s, %s);
            """
            cursor.execute(query, (asistencia, tareas, participacion, calificacion_final))
            conn.commit()
            cursor.close()

        # Re-entrenar el modelo con el nuevo dato
        entrenar_modelo()
        return jsonify({'message': 'Datos guardados y modelo actualizado.'}), 200

    except Exception as e:
        print(f"Error al guardar: {e}")
        return jsonify({'error': 'No se pudieron guardar los datos.'}), 500

# --- ENDPOINT PARA REALIZAR LA PREDICCIÓN Y GUARDAR ---
@app.route('/api/predict', methods=['POST'])
def predecir():
    try:
        datos = request.get_json()

        asistencia = float(datos['asistencia'])
        tareas = float(datos['tareas'])
        participacion = float(datos['participacion'])

        # Validaciones
        if not (0 <= asistencia <= 100 and 0 <= tareas <= 10 and 0 <= participacion <= 10):
            return jsonify({'error': 'Valores fuera de rango.'}), 400

        # 1. Realizar la predicción
        input_data = np.array([[asistencia, tareas, participacion]])
        prediccion = modelo.predict(input_data)[0]
        
        # Limitar nota entre 0 y 100
        prediccion = min(100.0, max(0.0, prediccion))
        
        # 2. GUARDAR EN LA BASE DE DATOS (¡Esto es lo nuevo!)
        # Así la predicción quedará registrada en el historial
        try:
            with closing(psycopg2.connect(**DB_CONFIG)) as conn:
                cursor = conn.cursor()
                query = """
                    INSERT INTO calificaciones (asistencia_porcentaje, tareas_completadas, participacion, calificacion_final)
                    VALUES (%s, %s, %s, %s);
                """
                cursor.execute(query, (asistencia, tareas, participacion, float(prediccion)))
                conn.commit()
        except Exception as e:
            print(f"Error al guardar en BD: {e}") 
            # (No detenemos el programa, solo avisamos en la consola negra)

        # 3. Devolver la respuesta al HTML
        return jsonify({
            'calificacion_predicha': round(prediccion, 2),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Ocurrió un error: {str(e)}'}), 500

# --- ENDPOINT PARA OBTENER EL HISTORIAL ---
@app.route('/api/historial', methods=['GET'])
def obtener_historial():
    try:
        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            query = "SELECT asistencia_porcentaje, tareas_completadas, participacion, calificacion_final FROM calificaciones ORDER BY fecha DESC LIMIT 50;"
            df_historial = sql.read_sql(query, conn)

        # Convertir a diccionario
        historial_json = df_historial.to_dict('records')
        return jsonify(historial_json)

    except Exception as e:
        print(f"Error historial: {e}")
        return jsonify({'error': 'No se pudo cargar el historial.'}), 500

# --- ENDPOINT PARA OBTENER LOS COMENTARIOS ---
@app.route('/api/comentarios', methods=['GET'])
def obtener_comentarios():
    try:
        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            query = "SELECT comentario FROM comentarios ORDER BY fecha DESC;"
            df_comentarios = sql.read_sql(query, conn)

        comentarios_json = df_comentarios.to_dict('records')
        return jsonify(comentarios_json)

    except Exception as e:
        print(f"Error comentarios: {e}")
        return jsonify({'error': 'No se pudieron cargar los comentarios.'}), 500

# --- FUNCIONES PARA MOSTRAR EL GRÁFICO ---
@app.route('/api/grafico', methods=['GET'])
def mostrar_grafico():
    global X, y
    try:
        if X.empty:
             return jsonify({'error': 'No hay datos suficientes para graficar.'}), 400

        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(8, 6))

        # CORRECCIÓN: Usar el nombre correcto de la columna del DataFrame
        # X es un DataFrame, así que usamos X['asistencia_porcentaje']
        # alpha=0.5 hace que sean 50% transparentes
        # s=100 hace los puntos un poco más grandes para verlos mejor
        ax.scatter(X['asistencia_porcentaje'], y, color='blue', alpha=0.5, s=100, label='Datos Reales')        
        # Para la línea de regresión, ordenamos los valores para que se vea bien
        X_sorted = X.sort_values(by='asistencia_porcentaje')
        ax.plot(X_sorted['asistencia_porcentaje'], modelo.predict(X_sorted), color='red', label='Regresión Lineal')

        ax.set_xlabel("Asistencia (%)")
        ax.set_ylabel("Calificación Final")
        ax.set_title("Impacto de la Asistencia en la Nota Final")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Guardar en memoria
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close(fig) # Cerrar para liberar memoria

        return jsonify({'grafico': img_base64})

    except Exception as e:
        print(f"Error gráfico: {e}")
        return jsonify({'error': 'No se pudo generar el gráfico.'}), 500

# --- ENDPOINT PARA GUARDAR UN NUEVO COMENTARIO ---
@app.route('/api/comentar', methods=['POST'])
def guardar_comentario():
    try:
        datos = request.get_json()
        texto_comentario = datos.get('comentario')

        if not texto_comentario:
             return jsonify({'error': 'El comentario no puede estar vacío'}), 400

        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            cursor = conn.cursor()
            # Solo guardamos el texto, la fecha se pone sola automática en la BD
            query = "INSERT INTO comentarios (comentario) VALUES (%s);"
            cursor.execute(query, (texto_comentario,))
            conn.commit()
            
        return jsonify({'message': 'Comentario guardado correctamente.'}), 200

    except Exception as e:
        print(f"Error al guardar comentario: {e}")
        return jsonify({'error': 'No se pudo guardar el comentario.'}), 500


# --- INICIO DEL SERVIDOR (Corregido) ---
if __name__ == '__main__':  # Se usan dos guiones bajos
    entrenar_modelo()
    print("Servidor iniciado en http://localhost:5000")
    app.run(debug=True)