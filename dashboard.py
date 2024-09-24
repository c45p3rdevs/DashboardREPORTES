import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import cv2
from flask import Flask, Response
import numpy as np

# Cargar el archivo Excel
file_path = 'Reportes_DGSP_2024_09_24.xlsx'  # Actualiza con la ruta correcta
df = pd.read_excel(file_path)

# Cargar el Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la app de Dash y Flask
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Simulación de asignación de puesto por detección de rostro
puestos = {1: "Desarrollador", 2: "Analista", 3: "Desarrollador", 4: "Administrativo"}

# Ruta de Flask para servir el video de la cámara
@server.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Función para capturar los fotogramas de la cámara y detectar rostros
def generate_frames():
    camera = cv2.VideoCapture(0)  # Captura desde la cámara 0 (predeterminada)
    
    while True:
        success, frame = camera.read()  # Lee un fotograma
        if not success:
            break
        else:
            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros en el fotograma
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Dibujar rectángulos alrededor de los rostros detectados
            for i, (x, y, w, h) in enumerate(faces, 1):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Simular la asignación de un puesto según el rostro detectado
                puesto = puestos.get(i, "Desconocido")
                
                # Mostrar el puesto sobre la imagen
                cv2.putText(frame, puesto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Convertir el fotograma a formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Usar el frame para el video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Layout del dashboard
app.layout = html.Div([
    html.H1("Dashboard Interactivo de Proyectos", style={'text-align': 'center'}),

    # Texto de introducción
    html.P("Este dashboard muestra información sobre proyectos, empleados y proveedores."),

    # Sección de captura de imagen con OpenCV y detección de rostro
    html.H2("Captura de Imagen con la Cámara y Detección de Rostro"),
    html.Img(src="/video_feed"),  # Muestra el video en la interfaz web con detección de rostro

    # Filtro por Tipo de Proyecto
    html.Div([
        html.Label("Selecciona Tipo de Proyecto:"),
        dcc.Dropdown(id="tipo_proyecto_filter",
                     options=[{'label': tipo, 'value': tipo} for tipo in df['TipoProyecto'].unique()],
                     multi=True,
                     value=df['TipoProyecto'].unique(),
                     style={'width': "80%"})
    ]),

    # Filtro por Proveedor
    html.Div([
        html.Label("Selecciona Proveedor:"),
        dcc.Dropdown(id="proveedor_filter",
                     options=[{'label': prov, 'value': prov} for prov in df['Proveedor'].unique()],
                     multi=True,
                     value=df['Proveedor'].unique(),
                     style={'width': "80%"})
    ]),

    # Gráfico dinámico de barras para Tipo de Proyecto
    dcc.Graph(id='tipo_proyecto_graph', figure={}),

    # Gráfico dinámico de barras para Proveedor
    dcc.Graph(id='proveedor_graph', figure={}),

    # Gráfico circular para Tipo de Proyecto
    dcc.Graph(id='tipo_proyecto_pie_graph', figure={})
])

# Callback para actualizar los gráficos con base en los filtros
@app.callback(
    [Output(component_id='tipo_proyecto_graph', component_property='figure'),
     Output(component_id='proveedor_graph', component_property='figure'),
     Output(component_id='tipo_proyecto_pie_graph', component_property='figure')],
    [Input(component_id='tipo_proyecto_filter', component_property='value'),
     Input(component_id='proveedor_filter', component_property='value')]
)
def update_graph(tipo_proyecto_selected, proveedor_selected):
    # Filtrar los datos según los filtros seleccionados
    filtered_df = df[(df['TipoProyecto'].isin(tipo_proyecto_selected)) &
                     (df['Proveedor'].isin(proveedor_selected))]

    # Gráfico 1: Proyectos por Tipo de Proyecto (Gráfico de barras)
    fig_tipo_proyecto = px.histogram(filtered_df, x="TipoProyecto",
                                     title="Proyectos por Tipo de Proyecto",
                                     labels={'TipoProyecto': 'Tipo de Proyecto', 'count': 'Número de Proyectos'},
                                     color="TipoProyecto")

    # Gráfico 2: Proyectos por Proveedor (Gráfico de barras)
    fig_proveedor = px.histogram(filtered_df, x="Proveedor",
                                 title="Proyectos por Proveedor",
                                 labels={'Proveedor': 'Proveedor', 'count': 'Número de Proyectos'},
                                 color="Proveedor")

    # Gráfico 3: Proyectos por Tipo de Proyecto (Gráfico circular)
    fig_tipo_proyecto_pie = px.pie(filtered_df, names='TipoProyecto', 
                                   title="Distribución de Proyectos por Tipo (Gráfico Circular)",
                                   hole=0.3)  # hole=0.3 para hacer un gráfico tipo 'donut'

    return fig_tipo_proyecto, fig_proveedor, fig_tipo_proyecto_pie

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
