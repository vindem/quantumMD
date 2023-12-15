import json

class Config(object):
    def __new__(cls, config_file):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
            f = open(config_file, "r")
            cls.execution_setup = json.loads(f.read())
            #print(cls.execution_setup)
            return cls.instance
        return cls.instance


