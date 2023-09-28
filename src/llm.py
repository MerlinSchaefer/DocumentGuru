from langchain.llms import CTransformers
import box
import yaml


# load config
with open('./config.yaml', 'r', encoding='utf8') as file:
    config = box.Box(yaml.safe_load(file))

def create_llm():
    llm = CTransformers(model = config.MODEL_BIN_PATH, 
                        model_type=config.MODEL_TYPE,
                        config={'max_new_tokens': config.MAX_NEW_TOKENS,
                                'temperature': config.TEMPERATURE}
                        )
    return llm