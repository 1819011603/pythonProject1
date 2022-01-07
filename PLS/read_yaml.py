import yaml #
import pathlib # version == 4.2b2
def readYaml(yaml_path):
    try:
        with open(yaml_path, encoding="utf-8") as f:
            conf = yaml.load(f)  # safe_load 由于安全性原因 已经禁止使用python object对象进行存储
        return conf
    except IOError:
        p = pathlib.Path(yaml_path)
        if not p.exists():
            print("the {} file is not exists!".format(yaml_path))
        else:
            print("read {} file is failed!".format(yaml_path))

def writeYaml(json,save_path):
    try:
        with open(save_path,"w",encoding="utf-8") as f:
            print("the yaml file saves in {}".format(save_path))
            yaml.dump(json,f)
    except IOError:
        print("the {}  yaml file save failed!".format(save_path))

if __name__ == '__main__':
    json = range(10)
    writeYaml(json, "2.yaml")
    print(readYaml("2.yaml"))