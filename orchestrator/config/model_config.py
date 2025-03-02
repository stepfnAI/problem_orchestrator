from sfn_blueprint import MODEL_CONFIG
DEFAULT_LLM_PROVIDER = 'openai'
DEFAULT_LLM_MODEL = 'gpt-4o-mini'

MODEL_CONFIG = {
    "data_mapper": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 500,
            "n": 1,
            "stop": None
        }
    },
    "join_suggester": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.4,
            "max_tokens": 800,
            "n": 1,
            "stop": None
        }
    },
    "clustering_agent" : {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.1,  # Lower temperature for code generation
            "max_tokens": 2000,  # Higher tokens for code generation
            "n": 1,
            "stop": None
        }
    },
    "clustering_strategy_selector": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 800,
            "n": 1,
            "stop": None
        }
    } 
}