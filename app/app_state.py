import logging
import threading

logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.cache = None
        self.model = None
        self.executor = None
        self.embedding_service = None
        self.initialized = False
        self.lock = threading.Lock()

    def initialize(self):
        """Inicializa todos los componentes de la aplicación"""
        with self.lock:
            if self.initialized:
                return
                
            try:
                logger.info("🚀 Inicializando estado de la aplicación...")
                
                # ... [código de inicialización existente] ...
                
                # Marcar como inicializado al final
                self.initialized = True
                logger.info("✅ Estado de la aplicación inicializado")
            except Exception as e:
                logger.critical(f"Error inicializando estado: {str(e)}", exc_info=True)
                if self.executor:
                    self.executor.shutdown(wait=False)
                raise RuntimeError(f"Fallo en inicialización: {str(e)}")

app_state = AppState()