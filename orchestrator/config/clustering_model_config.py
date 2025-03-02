from sfn_blueprint import MODEL_CONFIG
DEFAULT_LLM_PROVIDER = 'openai'
DEFAULT_LLM_MODEL = 'gpt-4o-mini'

MODEL_CONFIG["mapping_agent"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 300,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["aggregation_agent"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 500,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["clustering_agent"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,  # Lower temperature for code generation
        "max_tokens": 2000,  # Higher tokens for code generation
        "n": 1,
        "stop": None
    }
}


MODEL_CONFIG["clustering_strategy_selector"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 800,
        "n": 1,
        "stop": None
    }
} 