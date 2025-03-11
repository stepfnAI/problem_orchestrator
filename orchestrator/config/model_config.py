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
    },
    "feature_suggester": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 800,
            "n": 1,
            "stop": None
        },
    },
    "recommendation_explainer": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 800,
            "n": 1,
            "stop": None
        }
    },
    "approach_selector": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 800,
            "n": 1,
            "stop": None
        }
    },
    "data_type_suggester": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
            "n": 1,
            "stop": None
        }
    },
    "code_generator": {
        "openai": {
        "model": DEFAULT_LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 2000,
        "n": 1,
        "stop": None
    }
    },
    "categorical_feature_handler": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
            "n": 1,
            "stop": None
    }
    },
    "model_trainer": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.1,
            "max_tokens": 2000,
            "n": 1,
            "stop": None
        }
    },
    "model_selector": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
            "n": 1,
        "stop": None
    }
    },
    "data_splitter": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
    },
    "leakage_detector": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
    },
    "target_generator": {
        "openai": {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.1,  # Low temperature for code generation
            "max_tokens": 2000,  # Higher tokens for code generation
            "n": 1,
            "stop": None
        }
    }
} 

