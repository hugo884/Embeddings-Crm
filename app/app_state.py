import logging

logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.cache = None
        self.model = None
        self.executor = None
        self.embedding_service = None

app_state = AppState()