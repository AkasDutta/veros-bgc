import yaml
from yaml import SafeLoader

path = "C:\\Users\\Akash Dutta\\GitHub_AkasDutta\\Veros_plus\\veros-bgc\\veros_bgc\\setup\\bgc_phyonly_global_4deg\\npzd_physics_tracers.yml"
with open(path) as f:
    my_dox = yaml.load_all(f, Loader=yaml.Safeloader)
print(type(my_dox))
print(my_dox)