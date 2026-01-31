from pydantic_settings import BaseSettings, Field

class Settings(BaseSettings):
    """
    Clase de configuración que carga variables desde el entorno.
    Kubernetes inyectará estas variables desde ConfigMaps y Secrets.
    """
    # --- Configuración de Google OAuth ---
    google_client_id: str = Field(..., env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(..., env="GOOGLE_CLIENT_SECRET")
    redirect_uri: str = Field(..., env="REDIRECT_URI")

    # --- Configuración de IA ---
    ollama_base_url: str = Field(..., env="OLLAMA_BASE_URL")
    model_name: str = Field(..., env="MODEL_NAME")

    # --- Seguridad para Comunicación Interna ---
    internal_api_secret: str = Field(..., env="INTERNAL_API_SECRET")

    # La clase Config ya no es necesaria con pydantic-settings v2
    # La configuración se hace automáticamente a través de Field(env="...")
    
    class Config:
        # Esta línea es importante para que pydantic-settings sepa de dónde leer las variables.
        # En Kubernetes, las variables se inyectan directamente, así que esto es más para desarrollo local.
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Se crea una instancia global de settings para ser importada en toda la aplicación
settings = Settings()