import yaml

def load_configs(file_path='configs/configs.yaml'):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
