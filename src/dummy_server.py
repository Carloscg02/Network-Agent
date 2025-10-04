# dummy_server.py

from mcp.server.fastmcp import FastMCP

import asyncio

async def main():
    # Stateless server (no session persistence, no sse stream with supported client)
    mcp = FastMCP("MCPServer", stateless_http=True, json_response=True)
    
    @mcp.tool()
    def enviar_ticket_email(celda: str, hora: str, probabilidad: float, mensaje: str) -> str:
        """
        Simula enviar un correo de ticket para zonas intermedias.
        """
        print(f"[SERVER] Enviando ticket para la celda {celda} a la hora {hora} con prob={probabilidad:.2f}")
        print(f"[SERVER] Mensaje:\n{mensaje}\n")
        return f"Ticket simulado enviado a {celda} a las {hora}"
    
    # @mcp.prompt()
    # def generar_mensaje_ticket(celda: str, hora: str, probabilidad: float) -> str:
    #     """
    #     Crea un mensaje breve de ticket a enviar para un humano.
    #     """
    #     return f"Se detecta anomalía en la celda {celda} a la hora {hora} con probabilidad {probabilidad:.2f}. Acción requerida: revisión humana."

    print("Starting MCP server...")
    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())