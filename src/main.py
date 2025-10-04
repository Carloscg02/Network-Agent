import asyncio
import mcp_use
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import tomllib
from mcp_use import MCPAgent, MCPClient
from pathlib import Path
import pandas as pd
from classifier import AnomalyDetector

project_root = Path(__file__).resolve().parent.parent
pyproject_file = project_root / "pyproject.toml"
server_config = project_root / "servers.json"


prompt_template_ticket = ChatPromptTemplate.from_template(
    """
Eres un agente de red móvil. Se ha detectado una posible anomalía en la red. Has de generar un email de aviso utilizando MCP y los siguientes datos deben de llegar al servidor:
- celda = {CellName}
- hora = {Time}
- probabilidad = {prob:.2f}
- mensaje = "Posible fallo en la red."
"""
)

prompt_template_accion = ChatPromptTemplate.from_template(
    """
Eres un agente de red móvil encargado de ejecutar acciones automáticas en celdas cuando se detecta una anomalía en la región alta. 
Tu tarea es redactar un mensaje profesional, breve y directo, indicando la acción que se ha realizado y la justificación técnica.

Datos de red de la celda {CellName} a la hora {Time}:
- PRBUsageUL: {PRBUsageUL}
- PRBUsageDL: {PRBUsageDL}
- meanThr_DL: {meanThr_DL}
- meanThr_UL: {meanThr_UL}
- maxThr_DL: {maxThr_DL}
- maxThr_UL: {maxThr_UL}
- meanUE_DL: {meanUE_DL}
- meanUE_UL: {meanUE_UL}
- maxUE_DL: {maxUE_DL}
- maxUE_UL: {maxUE_UL}
- maxUE_UL+DL: {maxUE_DL_UL}

Acción a comunicar: {accion}

Normas heurísticas para referencia:
1. **Congestión UL**: PRBUsageUL > 5 * PRBUsageDL y meanThr_DL < 0.5 → reasignar BW a UL o derivar tráfico a otras celdas.
2. **Celda infrautilizada**: PRBUsageUL < 1 y PRBUsageDL < 1 y maxUE_UL+DL < 3 → incrementar BW asignado y ajustar broadcast.
3. **Fallo de contadores UL**: meanUE_UL < 0.05 y maxUE_UL > 0 → reiniciar medición de contadores UL.

Formato esperado:
[Anomalía en la celda {CellName} a la hora {Time}]
[Acción Realizada]: mensaje profesional que indique la acción ejecutada.
[Justificación]: breve explicación técnica basada en los datos de red y las normas heurísticas, dando valores concretos de los Datos de la red que respalden la acción en este caso. La explicación ha de ser coherente con la acción realizada.

No incluyas introducciones, agradecimientos ni comentarios adicionales. Solo redacta la acción y la justificación de manera profesional.
"""
)

def evaluar_regla(fila):
    """
    Evalúa la regla heurística sobre una fila de anomalía (zona alta)
    y devuelve la acción que el sistema ya ha tomado automáticamente.
    
    Devuelve:
    - regla_activa: descripción breve de la anomalía detectada
    - accion: acción ejecutada automáticamente
    """
    # Prioridad de reglas según impacto
    if fila["PRBUsageUL"] > 5 * fila["PRBUsageDL"] and fila["meanThr_DL"] < 0.5:
        return (
            "Congestión UL",
            "Se ha reasignado BW a UL y/o se ha derivado tráfico a otras celdas para reducir congestión."
        )

    if fila["PRBUsageUL"] < 1 and fila["PRBUsageDL"] < 1 and fila["maxUE_UL+DL"] < 3:
        return (
            "Celda infrautilizada",
            "Se ha incrementado el ancho de banda asignado a la celda y ajustado el broadcast para mejorar utilización."
        )

    if fila["meanUE_UL"] < 0.05 and fila["maxUE_UL"] > 0:
        return (
            "Fallo de contadores UL",
            "Se ha reiniciado la medición de contadores UL para corregir la anomalía."
        )

    # Si no cumple ninguna regla específica
    return (
        "Anomalía genérica",
        "Se ha aplicado un reajuste general de parámetros de la celda para corregir la anomalía."
    )

async def agente_red(df, detector, llm, agent_ticket, low_th, high_th):
    X = df[detector.features]
    probs, _ = detector.predict(X)

    for i, prob in enumerate(probs):
        cell_name = df.iloc[i]["CellName"]
        time_val = df.iloc[i]["Time"]

        if prob < low_th:
            print(f"[OK] [{time_val}] {cell_name} → Prob={prob:.2f} → No anomalía.")

        elif prob < high_th:
            print(f"[⚠] [{time_val}] {cell_name} → Prob={prob:.2f} → Zona intermedia, creando ticket...")
            prompt = prompt_template_ticket.format(
                prob=prob, CellName=cell_name, Time=time_val, region="intermedia"
            )
            result = await agent_ticket.run(prompt)
            print("Resultado del server:", result)

        else:  # Zona alta
            print(f"[🔥] [{time_val}] {cell_name} → Prob={prob:.2f} → Zona alta, acción automática...")

            fila = df.iloc[i]

            # Evaluar la regla en Python
            regla, accion = evaluar_regla(fila)

            # Formatear prompt para que el LLM solo redacte texto según la regla
            prompt = prompt_template_accion.format(
                prob=prob,
                CellName=cell_name,
                Time=time_val,
                region="alta",
                regla=regla,
                accion=accion,
                PRBUsageUL=fila["PRBUsageUL"],
                PRBUsageDL=fila["PRBUsageDL"],
                meanThr_DL=fila["meanThr_DL"],
                meanThr_UL=fila["meanThr_UL"],
                maxThr_DL=fila["maxThr_DL"],
                maxThr_UL=fila["maxThr_UL"],
                meanUE_DL=fila["meanUE_DL"],
                meanUE_UL=fila["meanUE_UL"],
                maxUE_DL=fila["maxUE_DL"],
                maxUE_UL=fila["maxUE_UL"],
                maxUE_DL_UL=fila["maxUE_UL+DL"],
            )

            response = llm.invoke(prompt)
            print(response.content)

async def main():

    # Cargar configuración
    with open(pyproject_file, "rb") as f:
        config = tomllib.load(f)

    #leer thresholds
    low_th = config["thresholds"]["low"]
    high_th = config["thresholds"]["high"]

    # Inicializar LLM
    use_chatgpt = config.get("openai", {}).get("use_chatgpt", False)
    if use_chatgpt:
        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
            model_name=config["openai"]["model_name"],
            temperature=config["openai"]["temperature"]
        )
    else:
        llm = ChatOllama(
            model=config["agent"]["llm_model"],
            base_url="http://localhost:11434" 
        )

    model_path = project_root / config["classifier"]["model_path"]
    detector = AnomalyDetector(path_model=model_path)

    # Conectar cliente MCP
    client = MCPClient.from_config_file(server_config)

    # Crear agente
    agent_ticket = MCPAgent(
        llm=llm,
        client=client,
        max_steps=5,
        system_prompt="Eres asistente de detección de anomalías en redes móviles."
    )

    print("🚀 Cliente MCP iniciado")

    data_path = project_root /"data/ML-MATT-CompetitionQT1920_test.csv"
    df_to_label = pd.read_csv(data_path)

    muestras = df_to_label.sample(n=7, random_state=None)

    await agente_red(muestras, detector, llm, agent_ticket, low_th, high_th)

    # Cerrar sesiones MCP
    await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())
    
