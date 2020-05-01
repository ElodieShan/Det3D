import inspect
#inspect模块是针对模块，类，方法，功能等对象提供些有用的方法。例如可以帮助我们检查类的内容，检查方法的代码，提取和格式化方法的参数等。
from det3d import torchie


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        #返回一个可以用来表示对象的可打印字符串，可以理解为java中的toString()。
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._module_dict.keys())
        )
        return format_str

    @property #把方法变成属性，通过self.name 就能获得name的值。
    def name(self):
        return self._name
    #因为没有定义它的setter方法，所以是个只读属性，不能通过 self.name = newname进行修改

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
            
        #在model文件夹下的py文件中，里面的class定义上面都会出现 @DETECTORS.register_module，意思就是将类当做形参，
        #将类送入了方法register_module()中执行。
        """
        if not inspect.isclass(module_class):#判断是否为类
            raise TypeError(
                "module must be a class, but got {}".format(type(module_class))
            )
        module_name = module_class.__name__
        if module_name in self._module_dict:#看该类是否已经登记在属性_module_dict中
            raise KeyError(
                "{} is already registered in {}".format(module_name, self.name)
            )
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        #对上面的方法，修改了名字，添加了返回值，即返回类本身
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy() #字典类型
    obj_type = args.pop("type") #字典pop：移除序列中key为‘type’的元素，并且返回该元素的值
    if torchie.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_type, registry.name)
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    #* *args是将字典unpack得到各个元素，分别与形参匹配送入函数中
    return obj_cls(**args)
