# Agente de Detección y Gestión de Anomalías en Red Móvil

## Overview

Este repositorio contiene un **agente inteligente** que utiliza un modelo de **machine learning** para detectar anomalías en una red móvil y tomar decisiones automáticas o generar tickets para revisión humana.

El modelo de clasificación ha sido entrenado previamente en el repositorio:

[Modelo de clasificación de anomalías](https://github.com/Carloscg02/Kaggle_Network_Anomaly_Detection.git)

Si se desea reentrenar el modelo, es posible descargarlo desde allí y realizar el entrenamiento localmente.

El agente se conecta a herramientas externas como Gmail haciendo uso del protocolo MCP siguiendo su implementación con [learning mcp](https://github.com/Marcus-Forte/learning-mcp.git). Se utiliza [mcp_use](https://docs.mcp-use.com/getting-started) para los clientes MCP y [FastMCP](https://gofastmcp.com/getting-started/welcome) los servidores MCP.


### Lógica

3 zonas de detección en función de la probabilidad de anomalía a que estima a partir de los parámetros de red de entrada:

- Zona normal (prob < 0.3): No se detecta anomalía, el sistema continúa en operación normal.
- Zona intermedia (0.3 ≤ prob < 0.5): Posible anomalía leve. El agente crea un ticket automático (todavía simulado) a través del servidor MCP (enviar_ticket_email) para revisión humana.
- Zona crítica (prob ≥ 0.5): Anomalía confirmada, se registra el evento y se recomienda acción inmediata.

**Versión inicial conectandose a un "dummy server" que tofavía no tiene funcionalidad real de conectividad con aplicaciones externas**


## Estructura del proyecto

````
agente_anomalias_red/
│
├── models/
│   └── rf_anomaly_detector.joblib
│
├── src/
│   ├── main.py               
│   ├── dummy_server.py       
│   └──  classifier.py       
│
├── servers.json          
└── pyproject.toml           
````

## Requisitos

- Python 3.11+
- Librerías especificadas en `pyproyect.toml`:
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `langchain`
  - `langchain-ollama`
  - `langchain-ollama`
  - `ollama`
---


## Uso

1. Clonar el repositorio.
2. Crear y activar un entorno virtual
```bash
conda create -n <nombre-entorno> python=3.11 -y
conda activate <nombre-entorno>
```
3. Instalar dependencias:
```bash
pip install -e .
```

### Si dispones de TOKEN de OpenAI y quieres usarlo

1. crea un archivo .env en la raiz del proyecto con:

```bash
OPENAI_API_KEY=tu_clave_de_openai_aqui
```

2. En `pyproyect.toml` activa use_chatgpt y ajusta la configuración que desees:

```bash
use_chatgpt = true
model_name = "gpt-3.5-turbo"
temperature = 0.2
max_tokens = 512
```

### Próximos pasos

- Añadir más modelos como Gemini.
- Conectividad real de MCPserver con aplicaciones externas